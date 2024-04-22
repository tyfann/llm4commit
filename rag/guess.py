
from guesslang import Guess


guess = Guess()

name = guess.language_name("""
diff --git a/.travis.yml b/.travis.yml @@ -6,7 +6,7 @@ env:\n- ROS_DISTRO=\"kinetic\"\n- ROS_DISTRO=\"melodic\"\ninstall:\n- - git clone --quiet --depth 1 https://github.com/ros-industrial/industrial_ci.git .industrial_ci -b master\n+ - git clone --quiet --depth 1 https://github.com/ros-industrial/industrial_ci.git .ci_config -b legacy\nscript:\n- - .industrial_ci/travis.sh\n+ - .ci_config/travis.sh\n
""")

print(name)
