# Training
 To train the tensor-domain network run the following code sequentially

 1. train_MAA_VVBP.py (fo the sagittal plane)
 2. train_MAA_VVBP2.py (for the coronal plane)

 After training that, we have to train Gsum module by running train_sinonet.py
 Then, we can run refinement module
 
 3. train_MAA_VVBP3.py

 # Test
  To test the image, run the test_scirpt.sh
  1. test_script.sh (for the sagittal plane)
  2. test_script2.sh (for the coronal plane)

 # Dataset prepearation
 In this code, we generated VVBP from the original CT image (see the BP_gen.py)
 The exclusive data processing files is in the data folder.
