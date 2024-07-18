P="BilinearProblem"
cd output
for f in * ;do
if [[ ! "$f" == *"$P"* ]];then
mv -- "$f" "$P""_$f"; fi done
cd ..




