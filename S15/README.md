
# Speeding up transformers training using pytorch lightning.


In this repo, we speed up training transformers on the english to french translation by:

1. Parameter Sharing.
2. One Cycle Policy. (Model trained for 30 epochs only).
3. Reducing the hidden layer in feed forward network from 1024 to 256.
4. Dynamic Padding.
5. Removing english sentences with > 150 tokens and french sentences with tokens > 150 + 10.


Training logs

![Screenshot 2023-09-08 at 4 14 01 pm](https://github.com/santule/ERA/assets/20509836/f11e7c11-a050-466e-aa76-7bc0832fdabe)

 
![Screenshot 2023-09-07 at 8 25 12 pm](https://github.com/santule/ERA/assets/20509836/9bb4f104-d157-42fa-b7e3-2c975d541334)

Sample Prediction from model

```
=============================================================
SOURCE - The ladder was finally fixed on the 28th of May.
TARGET - L'échelle fut définitivement installée le 28 mai.
PREDICTED - L ' échelle fut enfin fixé sur le 28 mai .
=============================================================
SOURCE - To complete his misfortune, this spout ended in a leaden pipe which bent under the weight of his body.
TARGET - Pour comble de malheur, cette gouttière était terminée par un tuyau de plomb qui fléchissait sous le poids de son corps.
PREDICTED - Pour terminer son malheur , cette trombe s ' était terminée dans une pipe de plomb qui se penchait sous le poids de son corps .
=============================================================
SOURCE - It was impossible to shut the door fast, as it scraped the floor.
TARGET - Il était impossible de fermer complètement la porte, qui frottait sur le plancher.
PREDICTED - Il était impossible de fermer la porte , en le parquet .
=============================================================
SOURCE - I wish to make you happy.
TARGET - Je veux vous rendre heureuse….
PREDICTED - Je veux vous faire heureux .

```
