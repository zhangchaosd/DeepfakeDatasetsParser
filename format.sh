for i in $*;
do
isort $i
done
for i in $*;
do
black -S -l 80 $i
done
for i in $*;
do
flake8 $i
done

# pip install isort black flake8
# chmod ugo+x format.sh