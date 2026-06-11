THIS_DIR=$(dirname $0)

echo ==============================
ls $THIS_DIR
echo ==============================
if [ $(pwd) != $(realpath $THIS_DIR) ]; then
	echo "Please do \"cd "$THIS_DIR"\""
else
    pwd=$(realpath $THIS_DIR)
	for example in $(ls -d */run.sh); do
        echo ==============================
        echo $example
        echo ------------------------------
        cd $(dirname $example)
        bash run.sh
        cd $pwd
        echo $example done
        echo ==============================
	done
fi
