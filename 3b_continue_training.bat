@echo off
REM ============================================================
REM  Continue an interrupted IC-LoRA run.
REM
REM  Loads the LATEST lora_weights_step_*.safetensors from
REM      G:\Lora\LTX-2\outputs\<RUN_NAME>\checkpoints
REM  and continues up to the total step count in 3_train_iclora.bat
REM  (STEPS=6000 means "stop at step 6000", not "6000 more steps").
REM
REM  All other settings come from 3_train_iclora.bat - edit them THERE,
REM  not here. Rank, optimizer and scheduler must stay the same as in
REM  the interrupted run, otherwise the trainer warns and starts from 0.
REM ============================================================
call "%~dp03_train_iclora.bat" resume
