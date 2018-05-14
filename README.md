# maptrace

Produce watertight polygonal vector maps by tracing raster images.

  - THIS IS NOT A GIS TOOL (the output is not georeferenced)

  - DO NOT USE THIS TOOL TO CIRCUMVENT COPYRIGHT RESTRICTIONS ON
    EXISTING MAPS (you will be judged in the next life)
  
  - DO NOT USE THIS TOOL ON MULTI-MEGAPIXEL IMAGES (because loops
    are really slow in Python)
    
On the other hand, it might come in handy to generate crisp display
images for presentations/articles from raster images you created
yourself.

## Requirements

 - Python 2 or 3
 - Numpy 1.10 or higher
 - Scipy 0.17 or higher
 - Pillow

## Image copyrights

The files in the [`images`](/images) directory are licensed as follows:

  - [`pa-2012-pres.png`](images/pa-2012-pres.png) [[source]](https://commons.wikimedia.org/wiki/File:Pennsylvania-2012_presidential_election-by_county.PNG), license [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed.en)
  
 - [`pa-counties.png`](images/pa-counties.png) [[source]](https://commons.wikimedia.org/wiki/File:US_Census_Bureau_Pennsylvania_County_Map.png), license [CC-BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/deed.en)
 
 - [`birds-head.png`](images/birds-head.png) adapted by E. Gasser from an unpublished map by [sil.org](https://www.sil.org/), all rights reserved

Maps created by `maptrace.py` from these input images (including the respective outputs in the `examples` directory) shall be considered derivative works for copyright purposes.

## Examples

The images in the [`examples`](/examples) directory were created by running the following commands:

    python maptrace.py -c images/pa-2012-pres.png -f open:cross,1 -n2
    python maptrace.py images/pa-counties.png -m 1000 -e1.42 -n4 -b4
    python maptrace.py -c images/birds-head.png -C8 -m40 -n8 -q5 -s8 -b16 -f dilate:box,2

(The last command might take a few minutes.)

## Software license

The `maptrace.py` program itself is provided under the [MIT license](/LICENSE).

