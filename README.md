# Kivy-Android-Camera

## Building locally

```bash
buildozer -v android debug

# build and run on connected phone
buildozer -v android debug deploy run logcat

# clean all
buildozer appclean
```

## Local recipes

Example: https://github.com/tshirtman/android_cython_example/blob/master/buildozer.spec
Cleaning: https://stackoverflow.com/questions/64754806/how-can-i-clean-a-buildozer-custom-recipe-build

buildozer android p4a -- clean_recipe_build sticzinger_ops --local-recipes `pwd`/libs/recipes


## Serve local files
python3 -m http.server 3333

## OpenCV SURF
https://stackoverflow.com/questions/64525121/sift-surf-set-opencv-enable-nonfree-cmake-solution-opencv-3-opencv-4

https://stackoverflow.com/questions/11172408/surf-vs-sift-is-surf-really-faster/11301126

https://stackoverflow.com/questions/17840661/is-there-a-way-to-make-numpy-argmin-as-fast-as-min

https://stackoverflow.com/questions/57810101/how-can-i-make-numba-access-arrays-as-fast-as-numpy-can

https://stackoverflow.com/questions/48357722/sum-array-of-arrays-in-cython