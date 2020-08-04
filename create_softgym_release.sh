RELEASE_DIR=../../corl_code_submission/softgym/
CUR_DIR=$(pwd)
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR
rsync -aP --exclude=data --exclude=.git --exclude=*.pkl  --exclude=*__pychache__* --exclude=.gitignore  --exclude=.idea --exclude=.gitmodule --exclude=PyFlex --exclude=PyFlex_old --exclude=PyFlexRobotics ./ $RELEASE_DIR
cd $RELEASE_DIR
rm -rf singularity_build.txt
rm -rf tests prepare.sh prepare_ec2.sh compile.sh release.sh create_softgym_release.sh create_softgym_files.sh
rm -rf experiments algorithms
rm -rf tests
rm -rf __pychache__

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf