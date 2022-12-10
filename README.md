# Model Fine-tuning Layer Prediction for Surgical Fine-tuning Tasks

**To replicate results:**
To run the program, the command is always: `python train.py`

In place of argument parsing, I used wandb to keep track of my runs so I just manually added the
prediction types and other training options in the sweep configurations at the top.

Training prediction options:
- `correct`: this always predicts the correct layer block for baseline
- `grad_mag`: this is my implementation of the grad RGN from the paper
- `grad_var`: gradient variance
- `path_norm`: path norm
- `layer_cmp`: same label layer output comparison
- `inner_loop_full_step_mann`: MANN inner model
- `inner_loop_full_step_mlp`: MLP inner model
