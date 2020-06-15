while :
do
  git add . > /dev/null 2>&1
  git commit -m "intermediate" > /dev/null 2>&1
  git push origin master > /dev/null 2>&1
	sleep 900
done