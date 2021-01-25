## Install

use the setup.py script to install the module:

```
python setup.py install
```
or
```
python setup.py install --user
```


## Usage

To generate a model transit light curve with or without a moon, use

```
from exomoon_characterizer.fitting import model_one_moon

...

model_transit_with_moon = model_one_moon(time, *transit_args, **transit_kwargs)

model_transit_witout_moon = model_no_moon(time, *transit_args, **transit_kwargs)
```


Use example/Kepler\_transit\_light\_curve to generate Kepler-like transit light curves for ML data sets.
