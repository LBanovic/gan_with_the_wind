while :
do
  git add . > /dev/null
  git commit -m "intermediate" > /dev/null
  git push origin master > /dev/null
	sleep 900
done