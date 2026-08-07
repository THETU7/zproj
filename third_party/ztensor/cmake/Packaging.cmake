# ----------------------------------------------------------------------------
# ztensor C++ packaging (install/export + CPack).
#
# Included only under -DZT_INSTALL_CXX=ON (top-level CMakeLists.txt). Produces
# lib/libztensor.a, include/ztensor/**, lib/cmake/ztensor/*.cmake for the
# release tarball. The gate is mandatory: scikit-build-core runs `cmake
# --install` for the wheel, and ungated install() rules would ship the C++
# artifacts inside every wheel.
# ----------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Record whether the .a was built with OpenMP so the installed config can
# re-link consumers to it (see ztensorConfig.cmake.in).
if(TARGET OpenMP::OpenMP_CXX)
    set(ZT_CONFIG_HAS_OPENMP 1)
else()
    set(ZT_CONFIG_HAS_OPENMP 0)
endif()

install(TARGETS ztensor ztensor_eigen
    EXPORT ztensorTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Recurse the whole include tree — include/ztensor/zt/eigen/EigenConvert.h is
# NOT in _zt_public_headers (src/CMakeLists.txt), so install(FILES) drops it.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ztensor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")

install(EXPORT ztensorTargets
    FILE ztensorTargets.cmake
    NAMESPACE zt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ztensor)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/ztensorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ztensorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ztensor)

# Static .a is arch-dependent -> do NOT pass ARCH_INDEPENDENT.
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ztensorConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ztensorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ztensorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ztensor)

set(CPACK_PACKAGE_NAME "ztensor")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A minimal C++17 CPU tensor library (static)")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/THETU7/ztensor")
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_VERBATIM_VARIABLES ON)
include(CPack)
