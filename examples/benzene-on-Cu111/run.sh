THIS_DIR=$(dirname $0)

echo ==============================
ls $THIS_DIR
echo ==============================
rm -rf ./xyz ./png
if [ $(pwd) != $(realpath $THIS_DIR) ]; then
	echo "Please do \"cd "$THIS_DIR"\""
else
	eval $(pixi shell-hook -e dev)
	which python
	pixi list | grep nequip
	adsorption-tune -cd . -cn config.yaml
fi
