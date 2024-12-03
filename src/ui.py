import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font
import os
import time
import numpy as np
import models
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
    



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
        
    #open a top-level window to inform the user as the models retrain
    def training_popup():
        popup = tk.Toplevel(window)
        popup.title("Training")
        msg = tk.Label(popup, text="Please wait while the model is retrained (this may take a while)...")
        msg.grid(row=0, column=0, padx=5,pady=5)
        
        #train all models with new imported data
        models.train_all()
        
        #when retraining is done, show this
        msg.config(text="The model has been retrained.")
        ok = tk.Button(popup, text="OK", command=popup.destroy)
        ok.grid(row=1, column=1, padx=5, pady=5)
        window.update_idletasks
        

    
    #method for importing dataset and retraining model
    def get_data():
        f = filedialog.askopenfilename(filetypes=[("CSV (Comma Separated Values)", "*.csv")])
        if f:
            models.load_data(file_path=f)
            training_popup()
           
    #method for adding a single student to the records 
    def add_one():
        student = [studye.get(),attende.get(),pare.get(),are.get(),exte.get(),sleepe.get(),preve.get(),
                   mote.get(),iace.get(),tutore.get(),fine.get(),tqe.get(),ste.get(),pine.get(),
                   physe.get(),lde.get(),pede.get(),dste.get(),gene.get(),scoree.get()]
        models.load_single(student)
        training_popup()
        
    #method for adding a set of student records
    def add_many():
        f = filedialog.askopenfilename(filetypes=[("CSV (Comma Separated Values)", "*.csv")])
        if f:
            models.load_data(file_path=f, replace=False)
            training_popup()
      
    #popup window in case you try to do predictions without training      
    def untrained_popup():
        popup = tk.Toplevel(window)
        popup.title("Hold on!")
        msg = tk.Label(popup, text="You haven't trained the models on data yet!")
        msg.grid(row=0, column=0, padx=5,pady=5)
        ok = tk.Button(popup, text="OK", command=popup.destroy)
        ok.grid(row=1, column=1, padx=5, pady=5)
        
    #popup window with individual student predictions
    def popup_predictions(lasso, ridge, poly):
        popup = tk.Toplevel(window)
        popup.title("Results")
        lres = "Model 1 (Lasso Regression): " + str(lasso)
        rres = "\nModel 2 (Ridge Regression): " + str(ridge)
        pres = "\nModel 3 (Polynomial Regression): " + str(poly)
        msg = tk.Label(popup, text=(lres + rres + pres))
        msg.grid(row=0, column=0, padx=5,pady=5)
        ok = tk.Button(popup, text="OK", command=popup.destroy)
        ok.grid(row=1, column=1, padx=5, pady=5)
    
    #popup window with batch predictions
    def plot_predictions(lasso, ridge, poly):
        popup = tk.Toplevel(window)
        popup.title("Results")
        # the figure that will contain the plot 
        figl = plt.figure(1)
        plt.hist(lasso)
        figr = plt.figure(2)
        plt.hist(ridge)
        figp = plt.figure(3)
        plt.hist(poly)
        
        lc = FigureCanvasTkAgg(figl, popup)
        rc = FigureCanvasTkAgg(figr, popup)
        pc = FigureCanvasTkAgg(figp, popup)
        
        lt = tk.Label(popup, text="Model 1 (Lasso Regression)")
        lt.grid(row=0, column=0)
        lplot = lc.get_tk_widget() 
        lplot.grid(row=1,column=0)
        rt = tk.Label(popup, text="Model 2 (Ridge Regression)")
        rt.grid(row=0, column=1)
        rplot = rc.get_tk_widget()
        rplot.grid(row=1,column=1)
        pt = tk.Label(popup, text="Model 3 (Polynomial Regression)")
        pt.grid(row=2, column=0)
        pplot = pc.get_tk_widget()
        pplot.grid(row=3,column=0)
    
        
    #predict a single student's scores
    def predict_one():
        if(models.has_data):
            student = [rstudye.get(),rattende.get(),rpare.get(),rare.get(),rexte.get(),rsleepe.get(),rpreve.get(),
                   rmote.get(),riace.get(),rtutore.get(),rfine.get(),rtqe.get(),rste.get(),rpine.get(),
                   rphyse.get(),rlde.get(),rpede.get(),rdste.get(),rgene.get()]
            lasso, ridge, poly = models.load_single(student, predicting=True)
            print(lasso, ridge, poly)
            popup_predictions(lasso, ridge, poly)
        else:
            untrained_popup()
            
    #predict batch of student scores
    def predict_many():
        if(models.has_data):
            f = filedialog.askopenfilename(filetypes=[("CSV (Comma Separated Values)", "*.csv")])
            lasso, ridge, poly = models.load_data(f, predicting=True)
            plot_predictions(lasso, ridge, poly)
        else:
            untrained_popup()
            
    
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
    lmhset = ("Low", "Medium", "High")
    ynset = ("Yes", "No")
    stset = ("Public", "Private")
    pnnset = ("Positive", "Negative", "Neutral")
    pedset = ("High School", "College", "Postgraduate")
    dstset = ("Near", "Moderate", "Far")
    genset = ("Male", "Female")
    
    studyp = tk.Label(t3, text="Hours Studied")
    studyp.grid(row=0, column=0, padx=20, pady=5)
    studye = tk.Entry(t3)
    studye.grid(row=0, column=1, padx=20, pady=5)
    attendp = tk.Label(t3, text="Attendance (%)")
    attendp.grid(row=1, column=0, padx=20, pady=5)
    attende = tk.Entry(t3)
    attende.grid(row=1, column=1, padx=20, pady=5)
    parp = tk.Label(t3, text="Parental Involvement")
    parp.grid(row=2, column=0, padx=20, pady=5)
    pare = ttk.Combobox(t3, width=18, values=lmhset)
    pare.grid(row=2, column=1, padx=20, pady=5)
    arp = tk.Label(t3, text="Access to Resources")
    arp.grid(row=3, column=0, padx=20, pady=5)
    are = ttk.Combobox(t3, width=18, values=lmhset)
    are.grid(row=3, column=1, padx=20, pady=5)
    extp = tk.Label(t3, text="Extracurricular Activities")
    extp.grid(row=4, column=0, padx=20, pady=5)
    exte = ttk.Combobox(t3, width=18, values=ynset)
    exte.grid(row=4, column=1, padx=20, pady=5)
    sleepp = tk.Label(t3, text="Hours of Sleep")
    sleepp.grid(row=5, column=0, padx=20, pady=5)
    sleepe = tk.Entry(t3)
    sleepe.grid(row=5, column=1, padx=20, pady=5)
    prevp = tk.Label(t3, text="Previous Scores")
    prevp.grid(row=6, column=0, padx=20, pady=5)
    preve = tk.Entry(t3)
    preve.grid(row=6, column=1, padx=20, pady=5)
    motp = tk.Label(t3, text="Motivation")
    motp.grid(row=7, column=0, padx=20, pady=5)
    mote = ttk.Combobox(t3, width=18, values=lmhset)
    mote.grid(row=7, column=1, padx=20, pady=5)
    iacp = tk.Label(t3, text="Internet Access")
    iacp.grid(row=8, column=0, padx=20, pady=5)
    iace = ttk.Combobox(t3, width=18, values=ynset)
    iace.grid(row=8, column=1, padx=20, pady=5)
    tutorp = tk.Label(t3, text="Tutoring Sessions")
    tutorp.grid(row=9, column=0, padx=20, pady=5)
    tutore = tk.Entry(t3)
    tutore.grid(row=9, column=1, padx=20, pady=5)
    finp = tk.Label(t3, text="Family Income Level")
    finp.grid(row=0, column=2, padx=20, pady=5)
    fine = ttk.Combobox(t3, width=18, values=lmhset)
    fine.grid(row=0, column=3, padx=20, pady=5)
    tqp = tk.Label(t3, text="Teacher Quality")
    tqp.grid(row=1, column=2, padx=20, pady=5)
    tqe = ttk.Combobox(t3, width=18, values=lmhset)
    tqe.grid(row=1, column=3, padx=20, pady=5)
    stp = tk.Label(t3, text="School Type")
    stp.grid(row=2, column=2, padx=20, pady=5)
    ste = ttk.Combobox(t3, width=18, values=stset)
    ste.grid(row=2, column=3, padx=20, pady=5)
    pinp = tk.Label(t3, text="Peer Influence")
    pinp.grid(row=3, column=2, padx=20, pady=5)
    pine = ttk.Combobox(t3, width=18, values=pnnset)
    pine.grid(row=3, column=3, padx=20, pady=5)
    physp = tk.Label(t3, text="Physical Activity Hours")
    physp.grid(row=4, column=2, padx=20, pady=5)
    physe = tk.Entry(t3)
    physe.grid(row=4, column=3, padx=20, pady=5)
    ldp = tk.Label(t3, text="Learning Disabilities")
    ldp.grid(row=5, column=2, padx=20, pady=5)
    lde = ttk.Combobox(t3, width=18, values=ynset)
    lde.grid(row=5, column=3, padx=20, pady=5)
    pedp = tk.Label(t3, text="Parental Education Level")
    pedp.grid(row=6, column=2, padx=20, pady=5)
    pede = ttk.Combobox(t3, width=18, values=pedset)
    pede.grid(row=6, column=3, padx=20, pady=5)
    dstp = tk.Label(t3, text="Distance from Home")
    dstp.grid(row=7, column=2, padx=20, pady=5)
    dste = ttk.Combobox(t3, width=18, values=dstset)
    dste.grid(row=7, column=3, padx=20, pady=5)
    genp = tk.Label(t3, text="Gender")
    genp.grid(row=8, column=2, padx=20, pady=5)
    gene = ttk.Combobox(t3, width=18, values=genset)
    gene.grid(row=8, column=3, padx=20, pady=5)
    scorep = tk.Label(t3, text="Exam Score")
    scorep.grid(row=9, column=2, padx=20, pady=5)
    scoree = tk.Entry(t3)
    scoree.grid(row=9, column=3, padx=20, pady=5)
    submit = tk.Button(t3, text="Submit", font=fnt, command=add_one)
    submit.grid(row=10,column=4,padx=10,pady=20)
    
    
    
    #add many students
    duadd = "Upload a new dataset to add onto the current set."
    da = tk.Label(t4, text=duadd, font=fnt)
    axpad = int((wdw-fnt.measure(duadd))/2)
    da.grid(row=0,column=0,padx=axpad,pady=50)
    
    #file select button
    adddata = tk.Button(t4, text="Choose a file...", font=fnt, command=add_many)
    adddata.grid(row=1,column=0,padx=10,pady=20)
    
    
    #predict a single score
    rstudyp = tk.Label(t5, text="Hours Studied")
    rstudyp.grid(row=0, column=0, padx=20, pady=5)
    rstudye = tk.Entry(t5)
    rstudye.grid(row=0, column=1, padx=20, pady=5)
    rattendp = tk.Label(t5, text="Attendance (%)")
    rattendp.grid(row=1, column=0, padx=20, pady=5)
    rattende = tk.Entry(t5)
    rattende.grid(row=1, column=1, padx=20, pady=5)
    rparp = tk.Label(t5, text="Parental Involvement")
    rparp.grid(row=2, column=0, padx=20, pady=5)
    rpare = ttk.Combobox(t5, width=18, values=lmhset)
    rpare.grid(row=2, column=1, padx=20, pady=5)
    rarp = tk.Label(t5, text="Access to Resources")
    rarp.grid(row=3, column=0, padx=20, pady=5)
    rare = ttk.Combobox(t5, width=18, values=lmhset)
    rare.grid(row=3, column=1, padx=20, pady=5)
    rextp = tk.Label(t5, text="Extracurricular Activities")
    rextp.grid(row=4, column=0, padx=20, pady=5)
    rexte = ttk.Combobox(t5, width=18, values=ynset)
    rexte.grid(row=4, column=1, padx=20, pady=5)
    rsleepp = tk.Label(t5, text="Hours of Sleep")
    rsleepp.grid(row=5, column=0, padx=20, pady=5)
    rsleepe = tk.Entry(t5)
    rsleepe.grid(row=5, column=1, padx=20, pady=5)
    rprevp = tk.Label(t5, text="Previous Scores")
    rprevp.grid(row=6, column=0, padx=20, pady=5)
    rpreve = tk.Entry(t5)
    rpreve.grid(row=6, column=1, padx=20, pady=5)
    rmotp = tk.Label(t5, text="Motivation")
    rmotp.grid(row=7, column=0, padx=20, pady=5)
    rmote = ttk.Combobox(t5, width=18, values=lmhset)
    rmote.grid(row=7, column=1, padx=20, pady=5)
    riacp = tk.Label(t5, text="Internet Access")
    riacp.grid(row=8, column=0, padx=20, pady=5)
    riace = ttk.Combobox(t5, width=18, values=ynset)
    riace.grid(row=8, column=1, padx=20, pady=5)
    rtutorp = tk.Label(t5, text="Tutoring Sessions")
    rtutorp.grid(row=9, column=0, padx=20, pady=5)
    rtutore = tk.Entry(t5)
    rtutore.grid(row=9, column=1, padx=20, pady=5)
    rfinp = tk.Label(t5, text="Family Income Level")
    rfinp.grid(row=0, column=2, padx=20, pady=5)
    rfine = ttk.Combobox(t5, width=18, values=lmhset)
    rfine.grid(row=0, column=3, padx=20, pady=5)
    rtqp = tk.Label(t5, text="Teacher Quality")
    rtqp.grid(row=1, column=2, padx=20, pady=5)
    rtqe = ttk.Combobox(t5, width=18, values=lmhset)
    rtqe.grid(row=1, column=3, padx=20, pady=5)
    rstp = tk.Label(t5, text="School Type")
    rstp.grid(row=2, column=2, padx=20, pady=5)
    rste = ttk.Combobox(t5, width=18, values=stset)
    rste.grid(row=2, column=3, padx=20, pady=5)
    rpinp = tk.Label(t5, text="Peer Influence")
    rpinp.grid(row=3, column=2, padx=20, pady=5)
    rpine = ttk.Combobox(t5, width=18, values=pnnset)
    rpine.grid(row=3, column=3, padx=20, pady=5)
    rphysp = tk.Label(t5, text="Physical Activity Hours")
    rphysp.grid(row=4, column=2, padx=20, pady=5)
    rphyse = tk.Entry(t5)
    rphyse.grid(row=4, column=3, padx=20, pady=5)
    rldp = tk.Label(t5, text="Learning Disabilities")
    rldp.grid(row=5, column=2, padx=20, pady=5)
    rlde = ttk.Combobox(t5, width=18, values=ynset)
    rlde.grid(row=5, column=3, padx=20, pady=5)
    rpedp = tk.Label(t5, text="Parental Education Level")
    rpedp.grid(row=6, column=2, padx=20, pady=5)
    rpede = ttk.Combobox(t5, width=18, values=pedset)
    rpede.grid(row=6, column=3, padx=20, pady=5)
    rdstp = tk.Label(t5, text="Distance from Home")
    rdstp.grid(row=7, column=2, padx=20, pady=5)
    rdste = ttk.Combobox(t5, width=18, values=dstset)
    rdste.grid(row=7, column=3, padx=20, pady=5)
    rgenp = tk.Label(t5, text="Gender")
    rgenp.grid(row=8, column=2, padx=20, pady=5)
    rgene = ttk.Combobox(t5, width=18, values=genset)
    rgene.grid(row=8, column=3, padx=20, pady=5)
    rsubmit = tk.Button(t5, text="Submit", font=fnt, command=predict_one)
    rsubmit.grid(row=10,column=4,padx=10,pady=20)
    
    
    #predict many scores
    prmany = "Upload a set of student data to predict their scores."
    prm = tk.Label(t6, text=prmany, font=fnt)
    prxpad = int((wdw-fnt.measure(duadd))/2)
    prm.grid(row=0,column=0,padx=prxpad,pady=50)
    
    #file select button
    getprs = tk.Button(t6, text="Choose a file...", font=fnt, command=predict_many)
    getprs.grid(row=1,column=0,padx=10,pady=20)
    
    window.mainloop()
    