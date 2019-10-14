#helper code to translate files from one directory into another
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# print(dir_path)

fp = open('./data/coco_500img.txt', 'r')
fp2 = open('./data/coco_500img2.txt', 'w')

for line in fp:
	# print(line)
	line = line.replace("../coco/images", dir_path+"/data/images")
	fp2.write(line)


fp.close()
fp2.close()