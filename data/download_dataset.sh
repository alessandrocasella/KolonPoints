#!/bin/bash
SCRIPTPATH=$(dirname $(readlink -f "$0"))
DATA_DIR="${SCRIPTPATH}"

cd $DATA_DIR

while IFS=" " read -r filename field; do
    field=${field%$'\r'}
    echo "------------------------"
    
    # Check if the unzip directory already exists
    unzip_dir="c3vd/${filename%.*}"
    if [ -d "$unzip_dir" ]; then
        echo "$filename has already been unzipped. Skipping..."
        continue
    fi
    
    echo "Downloading $filename...$field"
    wget  --wait 10 --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${field}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${field}" -O ${filename} && rm -rf /tmp/cookies. txt

    echo "Unzip $filename..."
    unzip -q "$filename" -d "c3vd/"

    echo "Removing unnecessary files in c3vd..."
    find c3vd/ -type f \( -name "*occlusion*" -o -name "*normals*" -o -name "*flow*" -o -name "*mesh*" \) -exec rm {} \;
    
    rm "$filename"
    
done < links.txt


