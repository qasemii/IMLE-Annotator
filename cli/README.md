#Instruction

Run the following command in your terminal to reach results in 
[this](https://arxiv.org/pdf/2209.04862.pdf) paper.

```bash
PYTHONPATH=. python3 ./cli/torch-cli.py -a 0 -e 20 -b 40 -k 3 -H 250 -m 350 
-K 5 -r 10 -M imle --aimle-target standard --imle-samples 1 --imle-noise gumbel 
--imle-input-temperature 1.0 --imle-output-temperature 1.0 --imle-lambda 1000.0 
--sst-temperature 0.0 --softsub-temperature 0.5 --ste-noise sog --ste-temperature 0.0 
--gradient-scaling --aimle-beta-update-momentum 0.0 --aimle-beta-update-step 0.0001 
--aimle-target-norm 1.0
```
