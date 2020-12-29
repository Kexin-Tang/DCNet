import os

def fileList(filePath, storePath):
	files = os.listdir(filePath)
	for filename in files:
		filename, _ = os.path.splitext(filename)
		with open(storePath, 'a') as f:
			f.write(filename)
			f.write('\n')

if __name__ == "__main__":
	rootPath = '/home/kxtang/DCNet_pytorch/data'
	fileName = 'BoundaryTestIMGS'

	filePath = rootPath + '/' + fileName

	storePath = rootPath + '/' + fileName + '/' + 'list.txt'

	fileList(filePath, storePath)
