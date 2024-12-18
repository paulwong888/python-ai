# accelerate launch -m axolotl.cli.train configs/bassic_train.yml
accelerate launch -m axolotl.cli.merge_lora configs/advance_train.yml
#accelerate launch -m axolotl.cli.train configs/official_train.yml