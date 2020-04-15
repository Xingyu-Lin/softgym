^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package fetch_depth_layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.7.12 (2017-08-02)
-------------------

0.7.11 (2017-07-31)
-------------------
* scoped 'isnan' with 'std'
  Fixes compilation errors while compiling with c++11 flags
* Contributors: Priyam Parashar

0.7.10 (2016-10-27)
-------------------
* updates to handle OpenCV3 changes
* Contributors: Michael Ferguson

0.7.9 (2016-07-26)
------------------

0.7.8 (2016-07-18)
------------------

0.7.7 (2016-06-20)
------------------

0.7.6 (2016-05-26)
------------------
* Allow FetchDepthLayer to send empty clouds
  This keeps the "current"-ness of observationbuffers up-to-date even when
  sensors return zero points due to being in a large open area.
* Contributors: Aaron Hoy

0.7.5 (2016-05-08)
------------------
* Fixed activate and deactivate to avoid errors
* Parameterizing inputs for ObservationBuffer to match parent layers.
* Contributors: Aaron Hoy, Alex Henning, Michael Ferguson

0.7.4 (2016-03-16)
------------------
* activate/deactivate for fetch depth layer
* Added transform filtering to depth layer depth data callback
* Contributors: Levon Avagyan

0.7.3 (2016-03-05)
------------------

0.7.2 (2016-02-24)
------------------
* Add option to clear with NANs
* Add ROS param clear_with_skipped_rays to re-enable clearing with edge rays
* Add ROS param to control size of skip region on edges
* Change min_clearing_height to -infinity
* Change max_clearing_height to +infinity
* Contributors: Aaron Hoy, Michael Ferguson

0.7.1 (2016-01-20)
------------------
* add parameters for topic names
* add support for static camera based on tf transform
* Improved local costmap clearing by considering all points for clearance.
* re-license fetch_depth_layer as BSD
* Contributors: Marek Fiser, Michael Ferguson

0.7.0 (2015-09-29)
------------------

0.6.2 (2015-07-30)
------------------

0.6.1 (2015-07-03)
------------------

0.6.0 (2015-06-23)
------------------

0.5.14 (2015-06-19)
-------------------

0.5.13 (2015-06-13)
-------------------

0.5.12 (2015-06-12)
-------------------

0.5.11 (2015-06-10)
-------------------

0.5.10 (2015-06-07)
-------------------

0.5.9 (2015-06-07)
------------------

0.5.8 (2015-06-07)
------------------

0.5.7 (2015-06-05)
------------------

0.5.6 (2015-06-04)
------------------

0.5.5 (2015-06-03)
------------------
* release fetch_depth_layer
* Contributors: Michael Ferguson
