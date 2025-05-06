# RT-DETR

RT-DETR is a Transformer-based model for real-time object detection. This project aims to generate ONNX models using the small model of RT-DETRv2 (RT-DETRv2-S) and achieve efficient inference.

## License

This project is released under the [Apache License 2.0](LICENSE).

## Requirements

To install the required packages:

```bash
pip install -r requirements.txt
```

## Examples

### Convert to ONNX

```bash
python convert_to_onnx.py
```

### Run Demo

```bash
python demo.py --image street.jpg
```

## License Details

This software is licensed under the Apache License 2.0. The main features of this license are:

- Commercial use is permitted
- Modification is permitted
- Distribution is permitted
- Patent use is permitted
- Private use is permitted

For detailed license information, please refer to the [LICENSE](LICENSE) file.

## Notes

- This project is intended for research purposes
- For commercial use, please comply with the license terms
- Contributions must comply with Apache License 2.0 