if [ ! -f glove.840B.300d.txt ]; then
  wget https://www.dropbox.com/s/ugj33d9is8msxyi/glove.840B.300d.zip?dl=1 -O glove.840B.300d.zip
  
  unzip glove.840B.300d.zip
fi
