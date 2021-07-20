
#download Datasets
[ ! -f "../Dataset/with_embeddings.zip" ] && echo "Downloading with_embeddings.zip" && gdown --id 1083YT2ku3BtGqcUSCvsMvagQruZDH22z -O "../Dataset/"

[ ! -d "../Dataset/with_embeddings/" ] && echo "Unziping with_embeddings.zip" && unzip ../Dataset/with_embeddings.zip -d ../Dataset/