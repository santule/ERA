import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torchmetrics
import warnings
warnings.filterwarnings("ignore")
import logging
from model import build_transformer
from config import get_config
from train import get_ds

logger = logging.getLogger("Transformer")
logger.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler(filename='prediction.log')
fileHandler.setLevel(level=logging.INFO)
logger.addHandler(fileHandler)

class LitTr(pl.LightningModule):
    def __init__(self,cfg,tokenizer_src,tokenizer_tgt):
        super().__init__()
        
        vocab_src_len = tokenizer_src.get_vocab_size()
        vocab_tgt_len = tokenizer_tgt.get_vocab_size()
        self.model = build_transformer(vocab_src_len, vocab_tgt_len, cfg['seq_len'], cfg['seq_len'], d_model=cfg['d_model'])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        self.BATCH_SIZE = cfg['batch_size']
        self.num_epochs = cfg['num_epochs']
        self.lr = cfg['lr']
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = cfg['seq_len']
        self.expected = []
        self.predicted = []
        self.initial_epoch = 0
        self.save_hyperparameters()
        self.train_loss = []


    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):

        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask  = batch['encoder_mask']
        decoder_mask  = batch['decoder_mask']

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.model.project(decoder_output) # ( b, seq_len, vocab_size)

        label = batch['label']

        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        self.train_loss.append(loss)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True,logger=True)
        return loss

    def casual_mask(self, size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return(mask == 0)

    def greedy_decode(self,source, source_mask):

        sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')

        encoder_output = self.model.encode(source, source_mask)
        decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source)

        while True:
            if decoder_input.size(1) == self.seq_len:
              break

            decoder_mask = self.casual_mask(decoder_input.size(1)).type_as(source_mask)

            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            prob = self.model.project(out[:,-1])
            _,next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item())], dim=1
            )

            if next_word == eos_idx:
              break

        return decoder_input.squeeze(0)

    def on_train_epoch_end(self):

        self.log('loss', torch.stack(self.train_loss).mean(), on_epoch=True, logger=True)
        print(f"Loss Mean - {torch.stack(self.train_loss).mean()}")
        self.train_loss.clear()

    def evaluate(self, batch, stage=None):

        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]

       
        model_out = self.greedy_decode(encoder_input, encoder_mask)

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]


        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        logger.info(f"SOURCE - {source_text}")
        logger.info(f"TARGET - {target_text}")
        logger.info(f"PREDICTED - {model_out_text}")
        logger.info("=============================================================")

        self.expected.append(target_text)
        self.predicted.append(model_out_text)


    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def on_validation_epoch_end(self):
        metric = torchmetrics.CharErrorRate()
        cer = metric(self.predicted, self.expected)
        self.log('validation_cer', cer, prog_bar=True, on_epoch=True, logger=True)


        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(self.predicted, self.expected)
        self.log('validation_wer', wer, prog_bar=True, on_epoch=True, logger=True)

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(self.predicted, self.expected)
        self.log('validation_bleu', bleu, prog_bar=True, on_epoch=True, logger=True)

        self.expected.clear()
        self.predicted.clear()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, eps = 1e-9)

        return optimizer


trainer = pl.Trainer(log_every_n_steps=1,
                     limit_val_batches=2,
                     check_val_every_n_epoch=1,
                     max_epochs=10, 
                     accelerator='auto',
                     devices='auto',
                     strategy='auto',
                     logger=[TensorBoardLogger("logs/", name="transformer-bilin")],
    )

def main():

    # config
    cfg = get_config()
    cfg['preload']=None
    cfg['num_epochs'] = 20
    cfg['batch_size'] = 16

    #### Loading Datasets
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(cfg)

    #### Loading the lightning module
    model = LitTr(cfg,tokenizer_src,tokenizer_tgt)
    trainer.fit(model, train_dataloader,val_dataloader)


if __name__ == "__main__":
  main()
