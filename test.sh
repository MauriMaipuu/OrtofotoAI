#!/bin/bash

# Path to ortophoto
input_ortho="Downloads/Orto/64991.tif"

# Output directory for tiles
output_dir="/home/user/Downloads/Orto/"

# Defining tile size
tile_size=800

# Split ortophoto into tiles
gdal2tiles.py -p raster -z 0-8 -w none $input_ortho $output_dir

# Process each tile
for tile in $(ls $output_dir*.tif); do
	echo "Processing $tile"

	# Get filename without extension
	filename=$(basename "$tile" .tif)

	# Defin output path for processed tile
	output_tile="${processed_dir}${filename}_processed.tif"

	#Process tile with PyTorch model
	python Downloads/model_testing_pytorch.py $tile $output_tile

	echo "Tile processed and saved to $output_tile"
done

echo "All tiles processed"
