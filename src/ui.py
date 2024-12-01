import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font
import os
import time
from models import *
    



#Launches the starting window for the application, centered on-screen.
def LaunchWindow():
    
    #current directory of this file
    cdir = os.path.dirname(os.path.abspath(__file__))

    #open a file with the default application
    def open_file(filepath):
        os.startfile(os.path.join(cdir,filepath))
    
    #open the help text file
    def open_help():
        open_file("HELP/HELP.txt")

    #open the about text file
    def open_about():
        open_file("ABOUT/ABOUT.txt")
    
    #method for importing dataset and retraining model
    def get_data():
        f = filedialog.askopenfile(mode='r', filetypes=[("CSV (Comma Separated Values)", "*.csv")])
        if f:
            data = f.read()
            newdata.grid_forget()
            duwt = "Please wait while the model is retrained (this may take a while)..."
            du.config(text=duwt)
            dxpad = int((wdw-fnt.measure(duwt))/2)
            du.grid(row=0,column=0,padx=dxpad,pady=50)
            #force tkinter to update the window's visuals
            window.update_idletasks()
            
            #insert call to retrain model with new file here
            #instead of making it look like something is happening with sleep
            time.sleep(5)
            
            duwt = "The model has been retrained."
            du.config(text=duwt)
            dxpad = int((wdw-fnt.measure(duwt))/2)
            du.grid(row=0,column=0,padx=dxpad,pady=50)
            window.update_idletasks()
            time.sleep(5)
            duwt = "Upload a new dataset to replace the current set."
            du.config(text=duwt)
            du.grid(row=0,column=0,padx=xpad,pady=50)
            newdata.grid(row=1,column=0,padx=10,pady=20)
            window.update_idletasks()
            
    def add_one():
        print("haha yeah")
        
    def add_many():
        f = filedialog.askopenfile(mode='r', filetypes=[("CSV (Comma Separated Values)", "*.csv")])
        if f:
            print("haha yeah")
            
    def predict_one():
        print("haha yeah")
    
    def predict_many():
        print("haha yeah")
    
    #create window
    window = tk.Tk()
    
    #get screen width and height and set geometry to center window on screen
    wdw = 800
    wdh = 600
    screenw = window.winfo_screenwidth()
    screenh = window.winfo_screenheight()
    crw = int(screenw/2 - wdw/2)
    crh = int(screenh/2 - wdh/2)
    window.geometry(f'{wdw}x{wdh}+{crw}+{crh}')
    window.title("Score Predictor App")
    
    #create sub-pages
    #each tab along the top handles a different function of the application
    s = ttk.Style()
    s.theme_create("topmenu", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {"configure": {"padding": [4, 10], "font" : ('Arial', '13', 'bold')},}})
    s.theme_use("topmenu")
    
    tabs = ttk.Notebook(window)
    tabs.pack()
    t1 = tk.Frame(tabs)
    tabs.add(t1, text="Home")
    t2 = tk.Frame(tabs)
    tabs.add(t2, text="Upload New Data")
    t3 = tk.Frame(tabs)
    tabs.add(t3, text="Add a Student")
    t4 = tk.Frame(tabs)
    tabs.add(t4, text="Add Many Students")
    t5 = tk.Frame(tabs)
    tabs.add(t5, text="Predict a Score")
    t6 = tk.Frame(tabs)
    tabs.add(t6, text="Predict Many Scores")
    
    
    #initial state (home)
    #calculate width of message on screen to center it, all other items in grid will center beneath
    fnt = font.Font(family="Arial", size=16)
    msg = "Welcome to the Score Predictor App."
    welcome = tk.Label(t1, text=msg, font=fnt)
    wxpad = int((wdw-fnt.measure(msg))/2)
    welcome.grid(row=0,column=0,padx=wxpad,pady=50)
    xpln_up = tk.Label(t1, text="Select an option above to get started.", font=fnt)
    xpln_up.grid(row=1,column=0,pady=0)
    spcr = tk.Label(t1, text="",font=fnt)
    spcr.grid(row=2,column=0,pady=20)
    
    #three buttons here. one opens a help text file, one opens an about file,
    #and one quits the program
    hlp = tk.Button(t1, text="Help", font=fnt, command=open_help)
    hlp.grid(row=3,column=0,padx=10,pady=10)
    abt = tk.Button(t1, text="About", font=fnt, command=open_about)
    abt.grid(row=4,column=0,padx=10,pady=10)
    quit = tk.Button(t1, text="Quit", font=fnt, command=window.destroy)
    quit.grid(row=5,column=0,padx=10,pady=10)
    
    
    #data upload tab
    #will only take csvs for simplicity
    duxp = "Upload a new dataset to replace the current set."
    du = tk.Label(t2, text=duxp, font=fnt)
    dxpad = int((wdw-fnt.measure(duxp))/2)
    du.grid(row=0,column=0,padx=dxpad,pady=50)
    
    #file select button
    newdata = tk.Button(t2, text="Choose a file...", font=fnt, command=get_data)
    newdata.grid(row=1,column=0,padx=10,pady=20)
    
    
    #add a single student
    studyp = tk.Label(t3, text="Hours Studied")
    studyp.grid(row=0, column=0, padx=20, pady=5)
    studye = tk.Entry(t3)
    studye.grid(row=0, column=1, padx=20, pady=5)
    attendp = tk.Label(t3, text="Attendance (%)")
    attendp.grid(row=1, column=0, padx=20, pady=5)
    attende = tk.Entry(t3)
    attende.grid(row=1, column=1, padx=20, pady=5)
    sleepp = tk.Label(t3, text="Hours of Sleep")
    sleepp.grid(row=2, column=0, padx=20, pady=5)
    sleepe = tk.Entry(t3)
    sleepe.grid(row=2, column=1, padx=20, pady=5)
    prevp = tk.Label(t3, text="Previous Scores")
    prevp.grid(row=3, column=0, padx=20, pady=5)
    preve = tk.Entry(t3)
    preve.grid(row=3, column=1, padx=20, pady=5)
    tutorp = tk.Label(t3, text="Tutoring Sessions")
    tutorp.grid(row=4, column=0, padx=20, pady=5)
    tutore = tk.Entry(t3)
    tutore.grid(row=4, column=1, padx=20, pady=5)
    physp = tk.Label(t3, text="Physical Activity Hours")
    physp.grid(row=5, column=0, padx=20, pady=5)
    physe = tk.Entry(t3)
    physe.grid(row=5, column=1, padx=20, pady=5)
    scorep = tk.Label(t3, text="Exam Score")
    scorep.grid(row=6, column=0, padx=20, pady=5)
    scoree = tk.Entry(t3)
    scoree.grid(row=6, column=1, padx=20, pady=5)
    submit = tk.Button(t3, text="Submit", font=fnt, command=add_one)
    submit.grid(row=7,column=3,padx=10,pady=20)
    
    
    #add many students
    duadd = "Upload a new dataset to add onto the current set."
    da = tk.Label(t4, text=duadd, font=fnt)
    axpad = int((wdw-fnt.measure(duadd))/2)
    da.grid(row=0,column=0,padx=axpad,pady=50)
    
    #file select button
    adddata = tk.Button(t4, text="Choose a file...", font=fnt, command=add_many)
    adddata.grid(row=1,column=0,padx=10,pady=20)
    
    
    #predict a single score
    studypr = tk.Label(t5, text="Hours Studied")
    studypr.grid(row=0, column=0, padx=20, pady=5)
    studyen = tk.Entry(t5)
    studyen.grid(row=0, column=1, padx=20, pady=5)
    attendpr = tk.Label(t5, text="Attendance (%)")
    attendpr.grid(row=1, column=0, padx=20, pady=5)
    attenden = tk.Entry(t5)
    attenden.grid(row=1, column=1, padx=20, pady=5)
    sleeppr = tk.Label(t5, text="Hours of Sleep")
    sleeppr.grid(row=2, column=0, padx=20, pady=5)
    sleepen = tk.Entry(t5)
    sleepen.grid(row=2, column=1, padx=20, pady=5)
    prevpr = tk.Label(t5, text="Previous Scores")
    prevpr.grid(row=3, column=0, padx=20, pady=5)
    preven = tk.Entry(t5)
    preven.grid(row=3, column=1, padx=20, pady=5)
    tutorpr = tk.Label(t5, text="Tutoring Sessions")
    tutorpr.grid(row=4, column=0, padx=20, pady=5)
    tutoren = tk.Entry(t5)
    tutoren.grid(row=4, column=1, padx=20, pady=5)
    physpr = tk.Label(t5, text="Physical Activity Hours")
    physpr.grid(row=5, column=0, padx=20, pady=5)
    physen = tk.Entry(t5)
    physen.grid(row=5, column=1, padx=20, pady=5)
    submit2 = tk.Button(t5, text="Submit", font=fnt, command=predict_one)
    submit2.grid(row=7,column=3,padx=10,pady=20)
    
    
    #predict many scores
    prmany = "Upload a set of student data to predict their scores."
    prm = tk.Label(t6, text=duadd, font=fnt)
    prxpad = int((wdw-fnt.measure(duadd))/2)
    prm.grid(row=0,column=0,padx=axpad,pady=50)
    
    #file select button
    getprs = tk.Button(t6, text="Choose a file...", font=fnt, command=predict_many)
    getprs.grid(row=1,column=0,padx=10,pady=20)
    
    window.mainloop()
    

#TESTING
LaunchWindow()