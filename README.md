# Quantum Brush

🔍 Hello! I am a creative image modification tool powered by quantum computing.
- Brush collection powered by quantum algorithms.
- Lightweight program which supports quantum simulation and hardware communication both.
- Work with high-res images with quantum backend to draw, modify and have fun!

👩🏻‍💻 Author: MOTH Quantum (This app is built with ❤️ by [Astryd Park](https://www.github.com/artreadcode)

---
---

📋 Contents
1. [Usage Instruction](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#usage-instruction)
2. [Installation Instruction](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#instllation-instruction)
3. [Examples](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#examples)
4. [Technical Stack](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#technical-stack)

---
---

## Usage Instruction
This application is tested with MacOS Sequoia (15.5) && Eclipse IDE (2025-03) && Python 3.11+ && OpenJDK 21.0.7 LTS. Technically, the application must support every machines (Windows, Linux and MacOS) with the suitable Java and Python versions. It requires OpenJDK and Python to execute, so they must be previously installed.

However, Luckily, the installer provides automatic OpenJDK + Python installation through `miniconda` thus you actually don’t need to do anything! `install.sh` creates condo environment `’quantumbrush’` and store all necessary libraries there. More of this is described under [Installation Instructions](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#instllation-instruction).

Quantum Brush is basically a graphics software powered by quantum-computing-imagination. There’s nothing which makes user experience difficult. The program has three windows. Treat them equally well.

1. Canvas

![The image of a canvas](https://private-user-images.githubusercontent.com/50163676/476522204-e0411fc0-f9ef-46d3-91f3-b786657c070d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ5MTE5ODQsIm5iZiI6MTc1NDkxMTY4NCwicGF0aCI6Ii81MDE2MzY3Ni80NzY1MjIyMDQtZTA0MTFmYzAtZjllZi00NmQzLTkxZjMtYjc4NjY1N2MwNzBkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODExVDExMjgwNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA4MGNiZGYzOTZhN2EyYTRjMTg0NTE1ZGVhMzgyZjU4NWIwNTE2ZTBkYmNjNjFhOTdmODYwZGFhNGQzM2M5ODcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.LV0DeIM-al9T0t26CgDWGAKFO7ZRnbx0OX3S9JUGPJM)
   This is the place where your image/canvas is displayed and you interact with it by `mouseClicked` and `mouseDragged`. `mouseClicked` will create a yellow dot. `mouseDragged` will create a red line. Look at the screenshot above.
   Those elements are crucial for quantum algorithms, so treat it nicely.

   ⚠️ CAVEAT: Because of this special requirements, after you work on other windows, you MUST click the title bar of the canvas or it will wrongly leave yellow dots. This might make quantum brush algorithms to misbehave so be careful!

   Now, what will you do if you want to do something with quantum brush algorithms? As a sidekick, there is a control panel. 
   
2. Control panel

![The image of a control panel](https://private-user-images.githubusercontent.com/50163676/476522309-5f6c6412-0c3f-495f-8b0b-bac223552f6f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ5MTIyNDgsIm5iZiI6MTc1NDkxMTk0OCwicGF0aCI6Ii81MDE2MzY3Ni80NzY1MjIzMDktNWY2YzY0MTItMGMzZi00OTVmLThiMGItYmFjMjIzNTUyZjZmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODExVDExMzIyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThmY2E0ZGJkZTdhYjYzMWIwM2Q1NDE4NmE3NGRmNmYwYzM5Yzk2YWU3NWFkMTM0M2Y2MDgyZDNiZjJlNzVjZDImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.e99s6lxbFLMWfFVuA9AjERB_ApzvjvNth9ysv_393ts)
	This window is for modifying parameters for quantum brush algorithms. It depends on which brush did you choose. For example, Heisenbrush (Ver.Continuous) here has a radius, lightness, saturation and strength. 

3. Stroke Manager

## Installation Instructions

## Examples

## Technical Stack