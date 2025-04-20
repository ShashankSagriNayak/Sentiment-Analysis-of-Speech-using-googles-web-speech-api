import customtkinter as ctk
from PIL import Image,ImageTk
import sounddevice as sd
from scipy.io.wavfile import write #used for storing the recorded audio into a file which can later be used for reading
from scipy.io.wavfile import read
import speech_recognition as sr #download name is SpeechRecognition import name is speech_recognition
import pyaudio #for microphone in  speech_recognition to work
import setuptools #for speech_recognition to work
from textblob import TextBlob as tb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

special_case_idioms_score = {
    "yeah right": -2,
    "kiss of death": -1.5,
    "hand to mouth": -2,
    "way to go": 3,
    "cut the mustard":-2,
# Positive idioms
    "ace in the hole": 2,
    "all ears": 1,
    "apple of my eye": 3,
    "at the top of one's game": 2.5,
    "ball is in your court": 1,
    "beat the odds": 2,
    "best of both worlds": 2.5,
    "blessing in disguise": 2,
    "break a leg": 2,
    "bring home the bacon": 1.5,
    "chip off the old block": 1,
    "cream of the crop": 2.5,

    "dream come true": 3,
    "easy as pie": 1.5,
    "fit as a fiddle": 2,
    "get the ball rolling": 1,
    "give the green light": 1.5,
    "golden opportunity": 2.5,
    "hit the nail on the head": 2,
    "in high spirits": 2.5,
    "jump for joy": 3,
    "keep your chin up": 1.5,
    "kill two birds with one stone": 2,
    "make hay while the sun shines": 1.5,
    "on cloud nine": 3,
    "on the ball": 2,
    "piece of cake": 1.5,
    "pull out all the stops": 2,
    "rise and shine": 1.5,
    "seal the deal": 2,
    "steal the show": 2.5,
    "the bee's knees": 2.5,
    "the best thing since sliced bread": 3,
    "the cream of the crop": 2.5,
    "the icing on the cake": 2.5,
    "the whole nine yards": 2,
    "time of your life": 3,
    "top-notch": 2.5,
    "walking on air": 3,

    # Neutral idioms
    "all in a day's work": 0,
    "back to square one": 0,
    "back to the drawing board": -0.5,
    "beat around the bush": -0.5,
    "break the ice": 0.5,
    "burn the midnight oil": 0,
    "by the book": 0,
    "call it a day": 0,
    "cross that bridge when you come to it": 0,
    "cut corners": -0.5,
    "every cloud has a silver lining": 1,
    "get your act together": 0,
    "go back to the drawing board": -0.5,
    "go with the flow": 0.5,
    "hit the books": 0,
    "hold your horses": 0,
    "it takes two to tango": 0,
    "keep an eye on": 0,
    "let the cat out of the bag": 0,
    "make ends meet": 0,
    "miss the boat": -0.5,
    "on the fence": 0,
    "play it by ear": 0,
    "pull yourself together": 0,
    "put all your eggs in one basket": -0.5,
    "read between the lines": 0,
    "see eye to eye": 0.5,
    "sit on the fence": 0,
    "sleep on it": 0,
    "stick to your guns": 0.5,
    "take the bull by the horns": 1,
    "the ball is in your court": 0,
    "think outside the box": 1,
    "tie the knot": 1,
    "time will tell": 0,
    "turn over a new leaf": 1,
    "under the weather": -0.5,

    # Negative idioms
    "add insult to injury": -2,
    "all bark and no bite": -1.5,
    "barking up the wrong tree": -1,
    "beat a dead horse": -1.5,
    "bite off more than you can chew": -1,
    "bite the bullet": -1,
    "blow a fuse": -2,
    "break the bank": -1.5,
    "burning bridges": -2,
    "bust your chops": -1.5,
    "cat got your tongue": -0.5,
    "caught red-handed": -2,
    "cry over spilled milk": -1,
    "cut it too close": -1,
    "down in the dumps": -2,
    "face the music": -1,
    "fall on deaf ears": -1.5,
    "get bent out of shape": -1.5,
    "get someone's goat": -1.5,
    "give someone the cold shoulder": -2,
    "go down in flames": -2.5,
    "hit the roof": -2,
    "in hot water": -1.5,
    "in over your head": -1.5,
    "jump down someone's throat": -2,
    "kick the bucket": -2,
    "lose your marbles": -1.5,
    "miss the mark": -1,
    "off the deep end": -2,
    "out of the frying pan into the fire": -2,
    "pain in the neck": -1.5,
    "raining cats and dogs": -0.5,
    "raise Cain": -1.5,
    "rock the boat": -1,
    "rub salt in the wound": -2.5,
    "run into a brick wall": -1.5,
    "scrape the bottom of the barrel": -1.5,
    "shoot yourself in the foot": -2,
    "sleep with the fishes": -3,
    "spill the beans": -0.5,
    "stab someone in the back": -3,
    "stick a fork in it": -1,
    "throw in the towel": -1.5,
    "throw someone under the bus": -2.5,
    "up the creek without a paddle": -2,
    "weather the storm": -1,
    "your goose is cooked": -2
}

def identify_idioms(speech_to_text_var, special_case_idioms_score):
    identified_idioms = []
    lower_text = speech_to_text_var.lower()
    for idiom in special_case_idioms_score:
        if idiom in lower_text:
            identified_idioms.append(idiom)
    return identified_idioms


def analyze_sentiment_with_idioms(speech_to_text_var, special_case_idiom_score):
    sentiment_machine = SentimentIntensityAnalyzer()

    # Identify idioms in the text
    found_idioms = identify_idioms(speech_to_text_var,special_case_idioms_score )

    # Perform sentiment analysis
    resultant_dictionary = sentiment_machine.polarity_scores(speech_to_text_var)

    # Adjust sentiment based on identified idioms
    for idiom in found_idioms:
        idiom_score = special_case_idioms_score[idiom] #dictionary will return the sentiment value
        resultant_dictionary['compound'] += idiom_score / 5  # Adjust compound score
        if idiom_score > 0:
            resultant_dictionary['pos'] += idiom_score / 5
        elif idiom_score < 0:
            resultant_dictionary['neg'] -= idiom_score / 5

    return resultant_dictionary, found_idioms


def speech_to_text():
    # initialize the recognizer
    brain_recognizer = sr.Recognizer()  # the brain of the sr which interprets what was said into the microphone

    with sr.Microphone() as ear_microphone:
        # used for opening the microphone(ear) when it comes out of the loop the microphone(ear) closes itself
        print("Say something!")
        audio_data = brain_recognizer.listen(ear_microphone)
        """recognizer will Listen from the ear_microphone as it it open for hearing processes and
           convert it to audio amplitude vector stored in audio_data
           analogously speaking the microphone is like the ear which is open for hearing process and closes when needed
           whereas the recognizer is ike the brain which interprets/decodes/identifies what is being heard by the ear"""
        try:
            # Recognize speech using google Web Speech API
            speech_to_text_var = brain_recognizer.recognize_google(audio_data=audio_data)
            """look recognize_google method is there its free so ot will not be shown to u but it has a 
               limit so use carefully and no disrespectful words allowed or get banned from using the service"""
            print("You said: " + speech_to_text_var)
            #blob_object=tb(speech_to_text_var)
            #sentiment=blob_object.sentiment
            #print(sentiment)
            resultant_dictionary, identified_idioms = analyze_sentiment_with_idioms(speech_to_text_var, special_case_idioms_score)

            print("Sentiment Analysis Results:")
            print(resultant_dictionary)
            print("\nIdentified Idioms:")
            print(identified_idioms)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.bar(list(resultant_dictionary.keys()), list(resultant_dictionary.values()), width=0.4)
            plt.title("Sentiment Analysis Scores")
            plt.xlabel("Sentiment Categories")
            plt.ylabel("Scores")
            plt.show()

            speech_to_text_label=ctk.CTkLabel(master=outer_frame,text=speech_to_text_var)
            speech_to_text_label.place(x=1,y=550)
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))



def record():
    """used for recording the audio here the audio will be recorded just for 5 seconds speak everything within the
    first 5 seconds or change the time_duration to match your need"""


    sample_rate = 44100  # number of samples in 1 second
    time_duration = 5  # total time to record the audio
    print("starting to record\n")
    recorded_voice = sd.rec(frames=sample_rate * time_duration, samplerate=sample_rate, channels=2)
    """creates an array of the magnitude audio amplitude at unit time and stores as a vector in recorded_voice"""
    sd.wait()  # wait till the audio is being recorded and u speak
    write(filename="trial1.wav", rate=sample_rate, data=recorded_voice)
    print("finished")



def playback():
    """used for playback the stored audio file"""
    sampling_rate, audio_to_be_played_vector = read("trial1.wav")

    # Play the audio
    sd.play(data=audio_to_be_played_vector, samplerate=sampling_rate)
    sd.wait()  # Wait until the audio has finished playing




#MAIN PROGRAM
background_image_recorder = ctk.CTkImage(dark_image=Image.open("C:/Users/Shashank Nayak/220968012/My Projects/OIP.jpg"),
                                        size=(35,35))
background_image_play = ctk.CTkImage(dark_image=Image.open("C:/Users/Shashank Nayak/220968012/My Projects/play.jpg"),
                                        size=(35,35))
#background_image = ctk.CTkImage(dark_image=Image.open("E:/Users/ANUP S PAI/PycharmProjects/sentiment_analysis/ui.jpg"),
                              #  size=(700, 600))


ctk.set_appearance_mode("dark") #can take 3 values that is light dark and system(mode which is used by your system)
ctk.set_default_color_theme("dark-blue") #can take 3 values blue,dark-bue,green

outer_frame=ctk.CTk() #setting the outside borders all functions will we put into this area
outer_frame.geometry("700x600") #setting the dimensions

bg_image_label = ctk.CTkLabel(outer_frame,text="")
#bg_image_label.pack()

button1=ctk.CTkButton(master=outer_frame,
                      #image=background_image_recorder,
                      text="Record",
                      corner_radius=50,
                      width=80,
                      height=80,
                      command=speech_to_text)
button1.place(x=330,y=270)
"""button2=ctk.CTkButton(master=outer_frame,
                      image=background_image_play,
                      text="",
                      corner_radius=10,
                      width=40,
                      height=40,
                      command=playback)
button2.place(x=350,y=300)
"""

#entry1=ctk.CTkEntry(master=outer_frame,width=400,height=40)
#entry1.place(x=1,y=550)
outer_frame.mainloop()
