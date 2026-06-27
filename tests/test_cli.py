from gesec_viewer.cli import build_parser


def test_cli_accepts_hardware_decode_argument():
    args = build_parser().parse_args(["--hw-decode", "off"])

    assert args.hw_decode == "off"
