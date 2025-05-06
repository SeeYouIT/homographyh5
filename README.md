#Rrquirements
pytorch 1.5
opencv

#
S1 pre trained HomographieNet model
S1.1 Generate pre training dataset
python b1.py
S1.2 Pre trained model
python train.py

S2 trains our model
S2.1 Generate training dataset
python b2.py
S2.2 Label Conversion
Change the parameter num=0000003 'in mark2h.cy, and then execute
python mark2h.py
Move the generated xxxxxx. txt file to/b/training and/b/validation
S2.2 Training Model
python train4paper.py

S3 usage model
python test4paper.py
