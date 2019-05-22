import czifile
import tiffile as tiff

IMAGEPATH = ''
image = czifile.imread(IMAGEPATH)
print(image.shape)
image = image.squeeze()
for i in range(image.shape[0]):
    with tiff.TiffWriter('/well/rittscher/users/achatrian/PTEN_TIFF/ch' + str(i) + '.tiff', bigtiff=True) as tifw:
        tifw.save(image[i], photometric='rgb')
