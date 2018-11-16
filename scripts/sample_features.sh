#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
python_=python3
else python_=python
fi

installs() {
    if [[ "$unamestr" == 'Linux' ]]; then
        lsb_release -a
        sudo rm /var/lib/apt/lists/lock
        sudo rm /var/cache/apt/archives/lock
        sudo rm /var/lib/dpkg/lock
        apt-get update
        sudo rm /var/lib/apt/lists/lock
        sudo rm /var/cache/apt/archives/lock
        sudo rm /var/lib/dpkg/lock
        yes | sudo dpkg --configure -a
        yes | sudo apt update
        yes | sudo apt install wget
        cd ~
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        chmod +x miniconda.sh
        bash miniconda.sh -b
        echo "try to find conda in ~"
        ls -lht ~/m*
        cuser=$(whoami)
#        av=$'export PATH=\"'
        av=$'export PATH=\"/home/ubuntu/miniconda3/bin:/home/'
        bv=$'/miniconda3/bin:$PATH\"'
        echo "$av$cuser$bv" > ~/.bash_profile
        echo "cat .bash_profile"
        cat ~/.bash_profile
        ls ~/miniconda3/bin
        source ~/.bash_profile
        echo "ls ~/miniconda3/bin"
        conda create -n p3 python=3
        source activate p3
        conda install -y numpy pandas=0.23.4 scikit-learn=0.20.0 cvxopt pathos pympler pymysql unidecode networkx \
            h5py seaborn gensim geopandas tqdm theano pymc3 pycparser nltk psutil
        conda install -y -c conda-forge python-igraph pytables python-levenshtein
        conda install -y -c dgursoy pywavelets
        pip install Distance
#        usname=`whoami`
#        python3 -m pip install pip numpy nose h5py pandas==0.23.4 scikit-learn==0.20.0 pympler Distance psutil
    fi
}

seed=$1
njobs=$2
mode=$3
verb=$4
trials=$5
subtrials=$6
estimators=$7

git_ab=alexander-belikov
git_kl=KnowledgeLab
package_name_bm=bm_support
package_name_dh=datahelpers

data_path=../
verb=INFO

setup_data() {
    tarfile=`ls *tar.gz`
    echo "Found tar.gz files" $tarfile "of size" $(du -smh $tarfile | awk '{print $1}')
    for tf in $tarfile; do
        tar xf $tf
    done
}

clone_repo() {
echo "starting cloning $1"
git clone https://github.com/$2/$1.git
echo "*** list files in "$1":"
ls -lht ./$1
echo "*** list files in "$1"/"$1 ":"
ls -lht ./$1/$1
cd ./$1
echo "starting installing $1"
python ./setup.py install
cd ..
}

exec_driver() {
cd ./$1/runners/
echo "starting exec_driver $1"
echo `which python`
python ./runner_find_features.py -s $data_path -d $data_path -s $seed\
                -p $njobs -m $mode -v $verb -t $trials -st $subtrials -e $estimators

cd ../..
}

post_processing() {
echo "*** all files sizes :"
ls -thor ~/*
}

# Install packages
installs
setup_data
# Clone repos from gh
clone_repo $package_name_dh $git_ab
clone_repo $package_name_bm $git_kl
# Execute the driver script
exec_driver $package_name_bm
# Prepare the results
post_processing
