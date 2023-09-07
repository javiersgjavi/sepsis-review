kaggle datasets download -d salikhussaini49/prediction-of-sepsis
unzip prediction-of-sepsis.zip -d prediction-of-sepsis

# create directory data/challenge/original if it does not exist
mkdir -p data/challenge/original

cp prediction-of-sepsis/Dataset.csv data/challenge/original/Dataset.csv

rm -r prediction-of-sepsis
rm prediction-of-sepsis.zip