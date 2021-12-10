# Build Instructions

1. `pip install -r requirements.txt`

sst.py usage (from sst.py --help)
```
usage: sst.py [-h] [-a ATTACK] [-l LAMDA] [-b BEAM] [-s SENTIMENT]
              [--beta BETA]

optional arguments:
  -h, --help            show this help message and exit
  -a ATTACK, --attack ATTACK
                        The type of attack by which to generate trigger
                        candidates
  -l LAMDA, --lamda LAMDA
                        lambda parameter for loss calculation. loss = loss +
                        lamda * gpt2_loss + beta
  -b BEAM, --beam BEAM  Beam size to use in getting best candidates. 1 if not
                        using beam search
  -s SENTIMENT, --sentiment SENTIMENT
                        Sentiment to filter on. 1 to flip positive to
                        negative; 0 to flip negative to positive
  --beta BETA           Beta parameter for loss calculation. loss = loss +
                        lamda * gpt2_loss + beta
  ```
