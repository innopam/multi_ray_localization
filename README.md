# Multi-ray-localization

Multi ray localization on OpenSfM or OpenDroneMap

This code is an algorithm that extracts the relative position of the coordinates of interest in a group form.

As a constraint, the coordinates of the input must form groups. And the way it receives json as input depends on OpenSfm.

Please see here.

https://github.com/mapillary/OpenSfM/blob/main/opensfm/io.py


# Installed

1. OpenDroneMap or OpenSfM

2. Requirements
```
    pip install plotly
    pip install pyproj
    pip install shapely
    pip install scipy
    pip install itertools
``` 
# Dataset

NAS4

/volume1/1_InternalCompany/Drone_dataset/PM2021002.제주작물모니터링/05_감귤나무/02_신효동 감귤나무/
#
Multi_Ray_local/Multi_Ray_Localization_cpu.py 를 사용하세요.

해당 python file은 opensfm에 여전히 의존하고 있습니다. 

하지만, 아직까지 issue가 존재합니다.

적은 양의 input point에 대해서는 잘 작동하지만, 많은 양의 input point에서는 알고리즘이 제 기능을 발휘하지 못합니다.

추가적인 알고리즘 개선 및 고속화 필요합니다.
