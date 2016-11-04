import os, sys
import numpy as np

#######
# GLOBAL VARS
#######

# how small to scale originals for NN inputs
# originals are 2048 x 1152 pixels
rescale_percentage = 14

# How to make PPM images:
	# http://askubuntu.com/questions/84409/converting-images-to-ppm-p3-using-convert-command
# How to use the convert tool:
	# http://www.imagemagick.org/script/convert.php
# Uses the Linux "convert" command to transform large .jpg files into small .ppm files with various effects applied
def convert_img(old_filename, new_filename):
	global rescale_percentage
	os.system("convert -auto-level -resize {}% -compress none {} {}".format(rescale_percentage,old_filename,new_filename))
	os.system("convert -auto-level -resize {}% -rotate 90 -compress none {} {}_1.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -rotate 180 -compress none {} {}_2.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -rotate 270 -compress none {} {}_3.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -rotate 90 -compress none {} {}_4.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -polaroid 10 -compress none {} {}_5.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -normalize -compress none {} {}_6.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -noise 3 -compress none {} {}_7.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -paint 4 -compress none {} {}_8.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -enhance -compress none {} {}_9.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))
	os.system("convert -auto-level -resize {}% -blue-shift 1.2 -compress none {} {}_10.ppm".format(rescale_percentage,old_filename,new_filename[:-4]))

# Converts the original set of image data into the altered set of image data
def convert_originals():
	suits = os.listdir("original/")
	dirs = os.listdir("./")
	class_label = 0
	example_id = 0
	if ("altered" in dirs):
		os.system("rm -r altered")
	os.system("mkdir altered")
	for suit in suits:													# form list of unique suits
		os.system("mkdir altered/{}".format(suit))						# make a directory for finished images
		cards = os.listdir("original/{}/".format(suit))					# form list of unique cards
		for card in cards:
			os.system("mkdir altered/{}/{}".format(suit,card))
			imgs = os.listdir("original/{}/{}/".format(suit,card))		# form list of images of the card
			for img in imgs:
				convert_img("original/{}/{}/{}".format(suit,card,img),"altered/{}/{}/{}_{}_{}_{}.ppm".format(suit,card,suit,card,class_label,example_id))
				example_id += 1
			class_label += 1

# Returns a numpy array of feature data, X, and an integer class value, y, for a given image
def parse_img(filename):
	file = open(filename,'r')
	name = filename.split("_")
	class_label = name[2]			# extract class label from filename
	count = 0
	feature_size = 0
	X = []
	for line in file:										# start reading in feature data
		if (count == 1):
			line = line.split(" ")
			feature_size = (int(line[0]) * int(line[1]))
		elif ((count != 0) & (count != 2)):
			line = np.asarray(line.split())
			for i in line:
				X.append(int(i))
		count += 1
	X = (np.asarray(X)).flatten()
	y = class_label
	return (X,y)

def main():
	# only need this call to convert_originals if you want to re-generate the input images (maybe at a different scale or something)
	# This will delete the current "altered" directory and all the images inside, and then reproduce all the altered training images
	#    and directories	
	convert_originals()

	# Read in all the image data; prepare it to be input for the NN
	card_features = []
	card_labels = []
	suits = os.listdir("altered/")
	for suit in suits:
		cards = os.listdir("altered/{}".format(suit))
		for card in cards:
			imgs = os.listdir("altered/{}/{}/".format(suit,card))
			for img in imgs:
				X,y = parse_img("altered/{}/{}/{}".format(suit,card,img))
				card_features.append(X)
				card_labels.append(y)
	print ("Data prepared for {} examples (each of size {} features) and {} corresponding class labels.".format(len(card_features),len(card_features[10]),len(card_labels)))


main()