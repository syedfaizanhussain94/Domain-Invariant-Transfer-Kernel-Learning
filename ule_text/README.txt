
	Dataset: ULE (Unsupervised Learning Example)

Original name: MNIST
Authors: Yann LeCun and Corinna Cortes
Domain: Handwritten digit recognition
Contact: Yann LeCun (http://yann.lecun.com)
Original resource url: http://yann.lecun.com/exdb/mnist/index.html

This dataset is a "toy example" for the Unsupervised and Transfer Learning Challenge. We provide all the data and the labels so the competitors can practice. 

If this were a regular challenge dataset, you would get only during phase I:
either (text format)
	ule_devel.data
	ule_valid.data
	ule_final.data
or (Matlab format)
	ule_devel.mat
	ule_valid.mat
	ule_final.mat
and (statistics on the data)
	ule.param

In phase II, you would get:
	ule_transfer.label

Here, because this is a toy example for practicing, you also get:
class names
	ule.classid
class labels
	ule_devel.label (this contains more labels than ule_transfer.label but has the same dimension)
	ule_valid.label
	ule_final.label
identities of the columns in the xxx.label files (the number indicates the class in classid)
	ule_devel.labelid
	ule_valid.labelid
	ule_final.labelid
	ule_transfer.labelid
id of the pattern in the original dataset
	ule_devel.dataid
	ule_valid.dataid
	ule_final.dataid
	

