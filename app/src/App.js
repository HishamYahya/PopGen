import './App.css';
import 'axios'
import axios from 'axios';
import { Component } from 'react'
import MidiPlayer from "midi-player-js";
import { Box, Button, Modal, Typography, Container, Checkbox, FormControlLabel, InputLabel, FormControl, Slider, CircularProgress, TextField, Grid, Chip, ListItem, Paper, IconButton, NativeSelect, Tooltip } from "@mui/material";
import { Piano } from "react-piano";
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import InfoIcon from '@mui/icons-material/Info';
import PauseCircleIcon from '@mui/icons-material/PauseCircle';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import Soundfont from 'soundfont-player'
import 'react-piano/dist/styles.css';
import fluidR3 from './fluidR3.json'


const API_PATH = 'https://api.popgen.app'

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      fileLoaded: false,
      playing: false,
      activeNotes: new Set(),
      ac: null,
      instrument: null,
      width: window.innerWidth,
      key: 'Any Key',
      temp: 1.2,
      tbar: 4,
      withChords: true,
      generating: false,
      chords: [],
      current_chord: -1,
      error: false
    }
    this.updateWindowDimensions = this.updateWindowDimensions.bind(this);

  }

  componentDidMount() {
    window.addEventListener('resize', this.updateWindowDimensions);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.updateWindowDimensions);
  }

  updateWindowDimensions() {
    this.setState({ width: window.innerWidth });
  }

  updateActiveNotes(event) {
    if (event.velocity === 0) {
      this.state.activeNotes.delete(event.noteNumber)
    } else {
      this.state.activeNotes.add(event.noteNumber)
    }
    if (this.chordPlayed > 0) {
      this.chordPlayed--
    } else {
      this.setState({})
    }
  }

  async initPiano() {
    this.state.ac = new AudioContext()

    this.state.instrument = this.state.instrument ?? await Soundfont.instrument(this.state.ac, this.state.instrument?.name ?? 'acoustic_grand_piano')

    this.player.on('midiEvent', (event) => {
      if (event.name === "Note on") {
        if (event.velocity === 0) {
          this.state.instrument.stop(event.noteNumber)
        } else {
          this.state.instrument.play(event.noteNumber)
        }
        this.updateActiveNotes(event)
      }
      if (event.name === 'Marker') {
        this.chordPlayed = 3
        this.setState(state => ({
          current_chord: state.current_chord + 1
        }))
      }
    });
    this.player.on('endOfFile', () => {
      this.stop()
    });
  
    this.setState(() => ({
      fileLoaded: true,
    }))

  }
  
  async generate() {
    this.setState(() => ({ generating: true }))
    if (!this.state.tbar)
      this.setState({ tbar: 4 })
    
    let res = await axios.get(API_PATH + '/generate', {
      params: {
        n_target_bar: this.state.tbar,
        temperature: this.state.temp,
        key: this.state.key,
        with_chords: this.state.withChords
      }
    })
    if (res.status !== 200) {
      this.setState({error: true, generating: false})
      console.log(res.statusText)
      return
    }
    if (this.state.playing) {
      this.stop()
    }

    this.events = res.data.events
    let buffer = new Uint8Array(res.data.buffer)
    this.player = new MidiPlayer.Player()
    this.player.buffer = buffer
    this.player.fileLoaded()
    await this.initPiano()

    let chords = []
    this.events.forEach(event => {
      const [name, value] = event.split('_')
      if (name === 'Chord') {
        chords.push(value)
      }
    });

    this.setState(() => ({ generating: false, chords }))


  }

  play() {
    if (this.state.fileLoaded) {
      this.player.play()
      this.setState(() => ({ playing: true }))
    }
  }

  pause() {
    if (this.state.fileLoaded) {
      this.player.pause()
      this.state.instrument.stop()
      this.setState(() => ({ playing: false }))
    }
  }

  stop() {
    if (this.state.fileLoaded) {
      this.player.stop()
      this.state.instrument.stop()
      this.player.resetTracks()
      this.setState(() => ({
        playing: false,
        current_chord: -1,
        activeNotes: new Set()
      }))
    }
  }

  async download() {
    let url = API_PATH + '/download?events=' + encodeURIComponent(this.events[0])
    for (let i = 1; i < this.events.length; i++) {
      url += '&events=' + encodeURIComponent(this.events[i])
    }

    window.open(url, '_blank')
  }

  async changeInstrument(name) {
    const ac = new AudioContext()
    const instrument = await Soundfont.instrument(this.state.ac, name)

    this.stop()

    this.setState(() => ({
      ac,
      instrument
    }))
  }

  render() {
    return (
      <Container component='main' width='xs'>
        <Box
          sx={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Box sx={{ mt: 1, flexDirection: 'column', display: 'flex', justifyContent: 'center', width: '80%' }}>
            {/* <Box sx={{display: 'flex', flexDirection: 'row', alignContent:'center'}}> */}
            <Grid container spacing={3} sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, alignContent: { xs: 'center' }, width: '100%' }}>
              <Grid container spacing={3} item xs={12} md={6} sx={{ flexDirection: 'row', display: 'flex', justifyContent: { xs: 'center', md: 'flex-start' }, alignContent: 'space-around', flexWrap: 'wrap' }}>
                <Grid item xs={7} width="100%">

                  <FormControl sx={{ width: '100%' }}>
                    <InputLabel id="key">Key</InputLabel>
                    <NativeSelect
                      id="key"
                      label="key"
                      fullWidth
                      value={this.state.key}
                      sx={{ width: '100%' }}
                      onChange={e => {
                        this.setState(() => ({ key: e.target.value }))
                      }
                      }
                    >
                      <option value='Any Key'>
                        Any Key
                    </option>
                      {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map(note => [
                        <option key={note + ':maj'} value={note + ':maj'}>{note + ':maj'}</option>,
                        <option key={note + ':min'} value={note + ':min'}>{note + ':min'}</option>
                      ]).flat()}
                    </NativeSelect>

                  </FormControl>
                </Grid>
                <Grid item xs={5}>

                  <TextField
                    label="Bars"
                    type='number'
                    variant="standard"
                    InputProps={{
                      inputProps: {
                        max: 16, min: 1
                      }
                    }}
                    fullWidth
                    value={this.state.tbar}
                    onChange={e => {
                      const newB = e.target.value
                      if ((newB > 0 && newB < 17))
                        this.setState(() => ({ tbar: newB }))
                      if (newB === '')
                        this.setState({ tbar: null })

                    }}
                  >

                  </TextField>
                </Grid>


              </Grid>

              <Grid item xs={12} md={6} sx={{ alignItems: 'center' }}>

                <InputLabel sx={{ mt: 2 }} id="temp">Temperature <Tooltip title="The higher the temperature, the more adventerous the model will be."><InfoIcon fontSize='inherit' /></Tooltip></InputLabel>
                <Slider
                  value={this.state.temp}
                  size='medium'

                  onChange={e => this.setState(() => ({ temp: e.target.value }))}
                  aria-label="Default"
                  valueLabelDisplay="auto" min={0} max={5} step={0.1} />

              </Grid>
            </Grid>
            <Box>

              <FormControlLabel

                control={<Checkbox
                  checked={this.state.withChords}
                  onChange={e => this.setState(() => ({ withChords: e.target.checked }))}
                />}
                label="Play Chords?"
                labelPlacement="end"
              />

            </Box>

            <Button
              variant='contained'
              sx={{ marginY: 1 }}
              onClick={() => this.generate()}
              disabled={this.state.generating}>
              {this.state.generating
                ? <Box sx={{display: 'flex', flexDirection: 'row', alignItems: 'center'}}>
                  <Typography>Generating... </Typography>
                  <CircularProgress />
                  </Box>
                : 'Generate'}
            </Button>
          </Box>
          <Paper
            sx={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              flexWrap: 'wrap',
              p: 2,
              m: 1,
            }}
          >
            <Box sx={{ mb: 2, mx: 1 }}>

              <InputLabel variant="standard" htmlFor="uncontrolled-native">
                Instrument
          </InputLabel>
              <NativeSelect
                fullWidth
                onChange={event => this.changeInstrument(event.currentTarget.value)}
                inputProps={{
                  name: 'instrument',
                  id: 'uncontrolled-native',
                }}
                value={this.state.instrument?.name || 'acoustic_grand_piano'}
              >
                {
                  fluidR3.map(instrument => {
                    let formatted = instrument.split('_')
                    formatted = formatted.map(word => {
                      if (isNaN(word)) {
                        word = word[0].toUpperCase() + word.slice(1)
                      }
                      return word
                    })
                    formatted = formatted.join(' ')
                    return <option key={instrument} value={instrument}>{formatted}</option>
                  })
                }
              </NativeSelect>
            </Box>
            <Box style={{ pointerEvents: 'none' }}>
              <Piano
                noteRange={{ first: 36, last: this.state.width > 600 ? 100 : 81 }}
                playNote={(midiNumber) => {
                  // Play a given note - see notes below
                }}
                stopNote={(midiNumber) => {
                  // Stop playing a given note - see notes below
                }}
                width={this.state.width * 0.75}
                activeNotes={Array.from(this.state.activeNotes)}

              />
            </Box>
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'row',
              }}
            >
              <IconButton onClick={() => this.stop()}>
                <StopCircleIcon fontSize='large' color='primary' />
              </IconButton>
              {
                this.state.playing
                  ? <IconButton onClick={() => this.pause()}>
                    <PauseCircleIcon fontSize='large' color='primary' />
                  </IconButton>
                  : <IconButton onClick={() => this.play()}>
                    <PlayCircleIcon fontSize='large' color='primary' />
                  </IconButton>

              }
              <Button onClick={() => this.download()}>
                Download MIDI
            </Button>
            </Box>
          </Paper>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'center',
              flexWrap: 'wrap',
              p: 0.5,
              m: 0,
            }}
            component="ul"
          >
            {this.state.chords.map((chord, index) => {
              return (
                <ListItem key={index} sx={{ maxWidth: 'fit-content' }}>

                  <Chip
                    label={chord}
                    color="info"
                    variant={this.state.current_chord === index ? 'filled' : 'outlined'}
                  />
                </ListItem>
              );
            })}
          </Box>
        </Box>
        <Modal
          open={this.state.error}
          onClose={() => this.setState({error: false})}
          aria-labelledby="modal-modal-title"
          aria-describedby="modal-modal-description"
        >
          <Box sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 400,
            bgcolor: 'background.paper',
            border: '2px solid #000',
            boxShadow: 24,
            p: 4,
          }}>
            <Typography id="modal-modal-title" variant="h6" component="h2">
              An error occurred while generating, please try again.
            </Typography>
            <Button onClick={() => this.setState({error: false})}>Close</Button>
          </Box>
        </Modal>
      </Container>

    );
  }
}

export default App;
