import pytest

from app.streaming import RangeNotSatisfiable, iter_file, parse_range, sniff_audio_extension

SIZE = 1000


def test_no_header_serves_full_file():
    assert parse_range(None, SIZE) is None
    assert parse_range("", SIZE) is None


def test_closed_range():
    assert parse_range("bytes=0-99", SIZE) == (0, 99)
    assert parse_range("bytes=100-199", SIZE) == (100, 199)


def test_open_ended_range():
    assert parse_range("bytes=500-", SIZE) == (500, 999)


def test_suffix_range():
    assert parse_range("bytes=-100", SIZE) == (900, 999)
    assert parse_range("bytes=-5000", SIZE) == (0, 999)  # larger than file: whole file


def test_end_is_clamped_to_file_size():
    assert parse_range("bytes=900-5000", SIZE) == (900, 999)


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=1000-1100", "bytes=-0", "bytes=5000-6000"])
def test_unsatisfiable(header):
    with pytest.raises(RangeNotSatisfiable) as exc:
        parse_range(header, SIZE)
    assert exc.value.size == SIZE


def test_empty_file_is_unsatisfiable():
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=0-10", 0)
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=-10", 0)


@pytest.mark.parametrize("header", ["bytes=0-99,200-299", "items=0-9", "bytes=abc-", "bytes=-", "bytes=50-10", "garbage"])
def test_ignored_headers_serve_full_file(header):
    assert parse_range(header, SIZE) is None


def test_iter_file_reads_exact_range_in_chunks(tmp_path):
    data = bytes(range(256)) * 20  # 5120 bytes
    path = tmp_path / "a.bin"
    path.write_bytes(data)

    chunks = list(iter_file(str(path), 100, 1123, chunk_size=300))
    assert b"".join(chunks) == data[100:1124]
    assert max(len(c) for c in chunks) <= 300  # bounded memory: never more than one chunk at a time
    assert len(chunks) == 4  # 1024 bytes / 300


def test_iter_file_whole_file(tmp_path):
    path = tmp_path / "b.bin"
    path.write_bytes(b"hello world")
    assert b"".join(iter_file(str(path), 0, 10)) == b"hello world"


def test_iter_file_stops_if_file_shrinks(tmp_path):
    path = tmp_path / "c.bin"
    path.write_bytes(b"abc")
    assert b"".join(iter_file(str(path), 0, 99)) == b"abc"


def test_sniff_audio():
    assert sniff_audio_extension(b"ID3\x04\x00\x00\x00\x00\x00\x00") == ".mp3"
    assert sniff_audio_extension(b"\xff\xfb\x90\x00") == ".mp3"
    assert sniff_audio_extension(b"RIFF\x24\x00\x00\x00WAVEfmt ") == ".wav"
    assert sniff_audio_extension(b"OggS\x00\x02") == ".ogg"
    assert sniff_audio_extension(b"fLaC\x00\x00") == ".flac"
    assert sniff_audio_extension(b"\x00\x00\x00\x20ftypM4A ") == ".m4a"
    assert sniff_audio_extension(b"MZ\x90\x00 not audio") is None
    assert sniff_audio_extension(b"") is None
