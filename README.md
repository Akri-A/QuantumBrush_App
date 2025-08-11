🔍 Hello! I am a creative image modification tool powered by quantum computing.
- Brush collection powered by quantum algorithms.
- Lightweight program which supports quantum simulation and hardware communication both.
- Work with high-res images with quantum backend to draw, modify and have fun!

👩🏻‍💻 Author: MOTH Quantum (This app is built with ❤️ by [Astryd Park](https://www.github.com/artreadcode)

---

📋 Contents
1. [Usage Instruction](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#usage-instruction)
2. [Installation Instruction](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#installation-instruction)
3. [Examples](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#examples)
4. [Technical Stack](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#technical-stack)

---

# Usage Instruction
This application is tested with MacOS Sequoia (15.5) && Eclipse IDE (2025-03) && Python 3.11+ && OpenJDK 21.0.7 LTS. Technically, the application must support every machines (Windows, Linux and MacOS) with the suitable Java and Python versions. It requires OpenJDK and Python to execute, so they must be previously installed.

However, Luckily, the installer provides automatic OpenJDK + Python installation through `miniconda` thus you actually don’t need to do anything! `install.sh` creates condo environment `’quantumbrush’` and store all necessary libraries there. More of this is described under [Installation Instructions](https://github.com/moth-quantum/quantum-brush-collab?tab=readme-ov-file#instllation-instruction).

Quantum Brush is basically a graphics software powered by quantum-computing-imagination. There’s nothing which makes user experience difficult. The program has three windows. Treat them equally well.

1. Canvas

![The image of Canvas](https://private-user-images.githubusercontent.com/50163676/476522204-e0411fc0-f9ef-46d3-91f3-b786657c070d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ5MTE5ODQsIm5iZiI6MTc1NDkxMTY4NCwicGF0aCI6Ii81MDE2MzY3Ni80NzY1MjIyMDQtZTA0MTFmYzAtZjllZi00NmQzLTkxZjMtYjc4NjY1N2MwNzBkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODExVDExMjgwNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA4MGNiZGYzOTZhN2EyYTRjMTg0NTE1ZGVhMzgyZjU4NWIwNTE2ZTBkYmNjNjFhOTdmODYwZGFhNGQzM2M5ODcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.LV0DeIM-al9T0t26CgDWGAKFO7ZRnbx0OX3S9JUGPJM)
   This is the place where your image/canvas is displayed and you interact with it by `mouseClicked` and `mouseDragged`. `mouseClicked` will create a yellow dot. `mouseDragged` will create a red line. Look at the screenshot above.
   Those elements are crucial for quantum algorithms, so treat it nicely.

   ⚠️ CAVEAT: Because of this special requirements, after you work on other windows, you MUST click the title bar of the canvas or it will wrongly leave yellow dots. This might make quantum brush algorithms to misbehave so be careful!

   Now, what will you do if you want to do something with quantum brush algorithms? As a sidekick, there is a control panel. 
   
2. Control panel

![The image of Control Panel](https://private-user-images.githubusercontent.com/50163676/476522309-5f6c6412-0c3f-495f-8b0b-bac223552f6f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ5MTIyNDgsIm5iZiI6MTc1NDkxMTk0OCwicGF0aCI6Ii81MDE2MzY3Ni80NzY1MjIzMDktNWY2YzY0MTItMGMzZi00OTVmLThiMGItYmFjMjIzNTUyZjZmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODExVDExMzIyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThmY2E0ZGJkZTdhYjYzMWIwM2Q1NDE4NmE3NGRmNmYwYzM5Yzk2YWU3NWFkMTM0M2Y2MDgyZDNiZjJlNzVjZDImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.e99s6lxbFLMWfFVuA9AjERB_ApzvjvNth9ysv_393ts)
	This window is for modifying parameters for quantum brush algorithms. It depends on which brush did you choose. For example, Heisenbrush (Ver.Continuous) here has a radius, lightness, saturation and strength. It also contains tiny descriptions of each brush.

    After creating & modifying each brush stroke, we must group those paths and level them up to a system called 'stroke'. When you press 'Create', the program will say you can open up Stroke Manager to run the quantum algorithms. It depends on your choice. You can make bunch of strokes BEFORE you run them on quantum simulations/hardware (in the near future) OR you can directly open up the manager window and run the algorithms. This program is designed to not interfere with creative workflow.

    Have you decided to move along to Stroke Manager?

3. Stroke Manager

	Open up Stroke Manager from *Tools* on the menu bar.

![The image of Stoke Manager](https://private-user-images.githubusercontent.com/50163676/476522241-00e21b8c-9c0c-4f1c-b98c-9866d594af2a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTQ5MTk3MjgsIm5iZiI6MTc1NDkxOTQyOCwicGF0aCI6Ii81MDE2MzY3Ni80NzY1MjIyNDEtMDBlMjFiOGMtOWMwYy00ZjFjLWI5OGMtOTg2NmQ1OTRhZjJhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwODExVDEzMzcwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRmNmEzYzU0NjVjODQ0MTMxOGU5YTIyZGRhYWIxYWIzY2FkMjBjOTZmOTlkMDczYzNmMmY1ZWNiNDcwMDJjZDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.55Byb2YnKPhIbjeZPfN7PAZuwUvmgAofutAbOq7mjbw)
	Here, you can see the list of ’strokes’ that you created. You can change the timeline of them, for example, run the stroke on the simulator 

# Installation Instructions
1. Look at the Release Tab right next to you.
2. Click the latest release version.
3. Click `install.sh` to download.
4. When it’s installed, open up ’Terminal’ on your MacOS computer. (If you don’t know where it is, press command+Space, type terminal and press Enter)
5. Open up the Finder window so that you can see the `install.sh` file.
6. Type `sh ` (Don’t forget to add Space!) on the Terminal window.
7. Drag `install.sh` from the Finder window and drop it on the Terminal window. You might see some weird path name is added next to `sh `. (e.g. `sh /astrydpark/download/install.sh`).
8. Press Enter and follow the instructions on the Terminal window.
   
	e.g. If the installer asks `{...}? (Y/y)`, you can press *y* and Enter!

9. After it install the program, you can choose whether the installer will set up the environment for you or not. Just press *y* for peaceful mind.

## (c.f.) Update Instructions
	You don't need to download another installation file. You can just simply browse through the `$HOME/quantumbrush` folder, and repeat the number 3-8 on the Terminal window, only for `update.sh` file this time!

# Examples

	You can see the tech-savvy details on our paper and understand deeply about quantum-powered creativity!
    
- Link: {LINK WILL BE ADDED}

# Technical Stack

- Format: Standalone Processing (Java) application for multiple OS
- Supported:
  
      ```
      (base) astrydpark@Astryds-MacBook-Pro ~ % java -version
          openjdk version "21.0.7" 2025-04-15 LTS
          OpenJDK Runtime Environment Temurin-21.0.7+6 (build 21.0.7+6-LTS)
          OpenJDK 64-Bit Server VM Temurin-21.0.7+6 (build 21.0.7+6-LTS, mixed mode, sharing)
      ```

- Works on Python 3.11+ (Automatic background: Recent version of Miniconda environment called ‘`quantumbrush`')

Project location: `$HOME/QuantumBrush`

Used IDE version: Eclipse 2025-03