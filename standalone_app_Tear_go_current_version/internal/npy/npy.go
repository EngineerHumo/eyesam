package npy

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"eyesam/standalone_app_Tear_go_current_version/internal/utils"
)

const (
	magicString = "\x93NUMPY"
)

var (
	shapeRegex = regexp.MustCompile(`\(\s*(\d+)\s*,\s*(\d+)\s*\)`)
	descrRegex = regexp.MustCompile(`'descr'\s*:\s*'([^']+)'`)
)

func WriteUint8(path string, mask utils.Mask) error {
	header, err := buildHeader("<u1", mask.Height, mask.Width)
	if err != nil {
		return err
	}
	buffer := bytes.NewBuffer(nil)
	if err := writeHeader(buffer, header); err != nil {
		return err
	}
	if _, err := buffer.Write(mask.Data); err != nil {
		return err
	}
	return os.WriteFile(path, buffer.Bytes(), 0o644)
}

func WriteFloat32(path string, mask utils.FloatMask) error {
	header, err := buildHeader("<f4", mask.Height, mask.Width)
	if err != nil {
		return err
	}
	buffer := bytes.NewBuffer(nil)
	if err := writeHeader(buffer, header); err != nil {
		return err
	}
	if err := binary.Write(buffer, binary.LittleEndian, mask.Data); err != nil {
		return err
	}
	return os.WriteFile(path, buffer.Bytes(), 0o644)
}

func ReadUint8(path string) (utils.Mask, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return utils.Mask{}, err
	}
	offset, descr, height, width, err := parseHeader(data)
	if err != nil {
		return utils.Mask{}, err
	}
	if descr != "<u1" {
		return utils.Mask{}, fmt.Errorf("unsupported dtype %s", descr)
	}
	expected := height * width
	if len(data[offset:]) < expected {
		return utils.Mask{}, fmt.Errorf("npy data size mismatch")
	}
	mask := utils.NewMask(width, height)
	copy(mask.Data, data[offset:offset+expected])
	return mask, nil
}

func ReadFloat32(path string) (utils.FloatMask, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return utils.FloatMask{}, err
	}
	offset, descr, height, width, err := parseHeader(data)
	if err != nil {
		return utils.FloatMask{}, err
	}
	if descr != "<f4" {
		return utils.FloatMask{}, fmt.Errorf("unsupported dtype %s", descr)
	}
	expected := height * width * 4
	if len(data[offset:]) < expected {
		return utils.FloatMask{}, fmt.Errorf("npy data size mismatch")
	}
	reader := bytes.NewReader(data[offset : offset+expected])
	mask := utils.NewFloatMask(width, height)
	if err := binary.Read(reader, binary.LittleEndian, mask.Data); err != nil {
		return utils.FloatMask{}, err
	}
	return mask, nil
}

func buildHeader(descr string, height, width int) (string, error) {
	if height <= 0 || width <= 0 {
		return "", fmt.Errorf("invalid shape")
	}
	header := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }", descr, height, width)
	return header, nil
}

func writeHeader(buffer *bytes.Buffer, header string) error {
	if _, err := buffer.WriteString(magicString); err != nil {
		return err
	}
	if err := buffer.WriteByte(1); err != nil {
		return err
	}
	if err := buffer.WriteByte(0); err != nil {
		return err
	}
	headerLen := len(header) + 1
	padLen := 16 - ((len(magicString) + 2 + 2 + headerLen) % 16)
	if padLen == 16 {
		padLen = 0
	}
	headerBytes := []byte(header + strings.Repeat(" ", padLen) + "\n")
	if err := binary.Write(buffer, binary.LittleEndian, uint16(len(headerBytes))); err != nil {
		return err
	}
	if _, err := buffer.Write(headerBytes); err != nil {
		return err
	}
	return nil
}

func parseHeader(data []byte) (int, string, int, int, error) {
	if len(data) < len(magicString)+4 {
		return 0, "", 0, 0, fmt.Errorf("invalid npy file")
	}
	if string(data[:len(magicString)]) != magicString {
		return 0, "", 0, 0, fmt.Errorf("invalid npy magic")
	}
	headerLen := int(binary.LittleEndian.Uint16(data[len(magicString)+2 : len(magicString)+4]))
	headerStart := len(magicString) + 4
	headerEnd := headerStart + headerLen
	if headerEnd > len(data) {
		return 0, "", 0, 0, fmt.Errorf("invalid header length")
	}
	header := string(data[headerStart:headerEnd])
	descrMatch := descrRegex.FindStringSubmatch(header)
	if len(descrMatch) < 2 {
		return 0, "", 0, 0, fmt.Errorf("missing dtype in header")
	}
	shapeMatch := shapeRegex.FindStringSubmatch(header)
	if len(shapeMatch) < 3 {
		return 0, "", 0, 0, fmt.Errorf("missing shape in header")
	}
	height, err := strconv.Atoi(shapeMatch[1])
	if err != nil {
		return 0, "", 0, 0, err
	}
	width, err := strconv.Atoi(shapeMatch[2])
	if err != nil {
		return 0, "", 0, 0, err
	}
	if strings.Contains(header, "True") {
		return 0, "", 0, 0, fmt.Errorf("fortran_order not supported")
	}
	return headerEnd, descrMatch[1], height, width, nil
}
