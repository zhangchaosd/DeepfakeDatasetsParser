
'''
Diverse Fake Face Dataset
real: FFHQ, CelebA, source of FF++
Identity and expression swap: FF++:I:FaceSwap, Deepfakes;E:Face2Face   and by DFL
Atrributes manipulation:FaceAPP, StarGAN. 4000 FFHQ and 2000 CelebA as source. for in FFHQ, generate 3 each face, two with a random filter one with multiple filters
for in CelebA, 40 by StarGAN. total 92k(4k*3+2k*40)
Entire face synthesis:PGGAN:200k, StyleGAN:100k ?


781,727real

1,872,007 fake 
randomly select a subset of
58,703 real
240,336 fake


       celebA  ffhq  ff++
train 
val
test



ffhq 10 000 999(少了10279) 9000
faceapp 6309+999+4501 with masks
pgganv1 9975+998+8970
pgganv2 9982+1000+8980
stargan 10 000 + 1000 + 21 913
stylegan_celeba 10 000 + 1000+9000
stylegan_ffhq 9 999 + 1000 + 8 997


test fake 135757
test real 25505

train fake 94836
train real 30204

vf 9743
vr 2994
'''
