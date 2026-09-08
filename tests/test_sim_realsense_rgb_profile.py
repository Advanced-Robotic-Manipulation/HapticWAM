"""Factory profile export never negotiates a stream or mutates camera state."""

from types import SimpleNamespace
import math

import pytest

from tools.sim.export_realsense_rgb_profile import export_profile, profile_fov


def fake_sdk(*, serial="944622074411", fps=15, fmt="rgb8", exposure_error=False):
    reads = []

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only export attempted to open/start/stop/change camera state")

    intrinsic = SimpleNamespace(
        width=640, height=480, fx=620.0, fy=615.0, ppx=309.0, ppy=247.0,
        model="distortion.modified_brown_conrady", coeffs=[.1, -.2, .003, .004, .05],
    )
    video = SimpleNamespace(width=lambda: 640, height=lambda: 480,
                            get_intrinsics=lambda: intrinsic)
    profile = SimpleNamespace(stream_type=lambda: "color", format=lambda: fmt,
                              fps=lambda: fps, stream_index=lambda: 0,
                              as_video_stream_profile=lambda: video)

    def get_option(option):
        reads.append(option)
        if exposure_error and option == "exposure":
            raise RuntimeError("camera busy")
        return {"enable_auto_exposure": 1, "enable_auto_white_balance": 1,
                "exposure": 200}[option]

    sensor = SimpleNamespace(
        get_stream_profiles=lambda: [profile],
        supports=lambda key: key in ("name", "enable_auto_exposure", "enable_auto_white_balance", "exposure"),
        get_info=lambda key: "RGB Camera", get_option=get_option,
        open=forbidden, close=forbidden, start=forbidden, stop=forbidden,
        set_option=forbidden,
    )
    info = {"name": "Intel RealSense D435", "serial_number": serial,
            "firmware_version": "test-firmware"}
    device = SimpleNamespace(supports=lambda key: key in info, get_info=info.__getitem__,
                             query_sensors=lambda: [sensor], hardware_reset=forbidden)
    sdk = SimpleNamespace(
        camera_info=SimpleNamespace(**{key: key for key in (
            "name", "serial_number", "firmware_version", "product_id", "usb_type_descriptor",
        )}),
        option=SimpleNamespace(**{key: key for key in (
            "enable_auto_exposure", "enable_auto_white_balance", "exposure",
        )}),
        stream=SimpleNamespace(color="color"), format=SimpleNamespace(rgb8="rgb8"),
        context=lambda: SimpleNamespace(query_devices=lambda: [device]),
        pipeline=forbidden, config=forbidden,
    )
    return sdk, reads, intrinsic


def test_export_queries_exact_rgb_intrinsics_and_options_without_camera_mutation():
    sdk, reads, _ = fake_sdk()
    result = export_profile(sdk)
    assert result["device"]["serial_number"] == "944622074411"
    assert result["profile"] == {
        "stream": "color", "format": "RGB8", "width": 640, "height": 480,
        "fps": 15, "stream_index": 0,
    }
    assert result["intrinsics"]["K"] == [[620, 0, 309], [0, 615, 247], [0, 0, 1]]
    assert result["intrinsics"]["distortion_coefficients"] == [.1, -.2, .003, .004, .05]
    assert result["options"]["enable_auto_white_balance"]["value"] == 1
    assert result["options"]["gain"] == {"supported": False, "value": None}
    assert reads == ["enable_auto_exposure", "enable_auto_white_balance", "exposure"]
    assert result["provenance"]["streams_started"] is False
    assert "not_proof_of_actively_negotiated" in result["provenance"]["status"]


@pytest.mark.parametrize("kwargs, error", [
    ({"serial": "another-camera"}, "Visible serials"),
    ({"fps": 30}, "No fallback selected"),
    ({"fmt": "bgr8"}, "No fallback selected"),
])
def test_wrong_device_or_mode_fails_without_fallback(kwargs, error):
    sdk, reads, _ = fake_sdk(**kwargs)
    with pytest.raises(RuntimeError, match=error):
        export_profile(sdk)
    assert reads == []


def test_no_devices_is_an_explicit_error():
    sdk, _, _ = fake_sdk()
    sdk.context = lambda: SimpleNamespace(query_devices=lambda: [])
    with pytest.raises(RuntimeError, match=r"Visible serials: \[\]"):
        export_profile(sdk)


def test_busy_option_is_reported_without_discarding_calibration():
    sdk, _, _ = fake_sdk(exposure_error=True)
    result = export_profile(sdk)
    assert result["options"]["exposure"] == {
        "supported": True, "value": None, "read_error": "camera busy",
    }
    assert result["intrinsics"]["fx"] == 620


def test_fov_preserves_off_center_principal_point_and_prefers_sdk():
    sdk, _, intrinsic = fake_sdk()
    fov, method = profile_fov(intrinsic, sdk)
    expected = math.degrees(math.atan(309.5 / 620) + math.atan(330.5 / 620))
    assert fov[0] == pytest.approx(expected)
    assert abs(fov[0] - math.degrees(2 * math.atan(320 / 620))) > .005
    assert method == "librealsense_rs2_fov_equivalent_pinhole_formula"
    sdk.rs2_fov = lambda value: [61.1, 42.2] if value is intrinsic else pytest.fail("Wrong intrinsics")
    assert profile_fov(intrinsic, sdk) == ([61.1, 42.2], "pyrealsense2.rs2_fov")
