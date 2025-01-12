axolotl inference configs/advance_train.yml \
    --lora-model-dir="./models/Llama3_Storyteller" \
    --gradio \
    --load-in-8bit true \
    --gradio-server-name 0.0.0.0