from ovrolwasolar import solar_pipeline as lwasp
import pytest
import os, sys, glob



caltable = '/data07/peijinz/testdir/testdata/caltables/20240517_100405_55MHz.bcal'
sun_slow_ms = '/data07/peijinz/testdir/testdata/slow/20240519_173002_55MHz.ms'

lwasp.image_ms_quick( solar_ms= sun_slow_ms , bcal=caltable, num_phase_cal=0, 
                     auto_pix_fov=True,
                     num_apcal=1 , logging_level='debug')

