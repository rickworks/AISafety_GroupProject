# AISafety_GroupProject
For dumping useful bits of code for the Cambridge Engineering Safe AI seminar group project

If you happen to be running on ubuntu 16.04 or later, here is how to install the stuff you need (requires python >= 3.5):
```
python -m venv gym_env
source gym_env/bin/activate
pip install gym gym-retro
```
to get the atari games (including Pong), install the ROMs file from http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html, then navigate to wherever that is in the terminal (make sure gym_env is still active) and run
```
sudo apt install unrar
unrar e Roms.rar
python -m retro.import
```
Get in touch if this doesn't work.

If you're running on something else, and struggle to find the equivilant commands, get in touch and I can help out.
