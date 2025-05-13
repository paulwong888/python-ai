accelerate launch -m axolotl.cli.train configs/deepspeed_train.yml --deepspeed configs/deepspeed/zero3.json
# accelerate launch -m axolotl.cli.train configs/fsdp_train.yml
#accelerate launch -m axolotl.cli.train configs/official_train.yml