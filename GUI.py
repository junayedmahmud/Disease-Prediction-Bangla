from tkinter import *
import tkinter as tk
import re
import main

root = Tk()


class AutocompleteEntry(tk.Entry):
    def __init__(self, autocompleteList, *args, **kwargs):

        self.listboxLength = 0
        self.parent = args[0]

        # Custom matches function
        if 'matchesFunction' in kwargs:
            self.matchesFunction = kwargs['matchesFunction']
            del kwargs['matchesFunction']
        else:
            def matches(fieldValue, acListEntry):
                pattern = re.compile(
                    '.*' + re.escape(fieldValue) + '.*', re.IGNORECASE)
                return re.match(pattern, acListEntry)

            self.matchesFunction = matches

        # Custom return function
        if 'returnFunction' in kwargs:
            self.returnFunction = kwargs['returnFunction']
            del kwargs['returnFunction']
        else:
            def selectedValue(value):
                print(value)

            self.returnFunction = selectedValue

        tk.Entry.__init__(self, *args, **kwargs)
        # super().__init__(*args, **kwargs)
        self.focus()

        self.autocompleteList = autocompleteList

        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = tk.StringVar()

        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.moveUp)
        self.bind("<Down>", self.moveDown)
        self.bind("<Return>", self.selection)
        self.bind("<Escape>", self.deleteListbox)

        self.listboxUp = False

    def deleteListbox(self, event=None):
        if self.listboxUp:
            self.listbox.destroy()
            self.listboxUp = False

    def select(self, event=None):
        if self.listboxUp:
            index = self.listbox.curselection()[0]
            value = self.listbox.get(tk.ACTIVE)
            self.listbox.destroy()
            self.listboxUp = False
            self.delete(0, tk.END)
            self.insert(tk.END, value)
            self.returnFunction(value)

    def changed(self, name, index, mode):
        if self.var.get() == '':
            self.deleteListbox()
        else:
            words = self.comparison()
            if words:
                if not self.listboxUp:
                    self.listboxLength = len(words)
                    self.listbox = tk.Listbox(self.parent,
                                              width=self["width"], height=self.listboxLength)
                    self.listbox.bind("<Button-1>", self.selection)
                    self.listbox.bind("<Right>", self.selection)
                    self.listbox.place(
                        x=self.winfo_x(), y=self.winfo_y() + self.winfo_height())
                    self.listboxUp = True
                else:
                    self.listboxLength = len(words)
                    self.listbox.config(height=self.listboxLength)

                self.listbox.delete(0, tk.END)
                for w in words:
                    self.listbox.insert(tk.END, w)
            else:
                self.deleteListbox()

    def selection(self, event):
        if self.listboxUp:
            self.var.set(self.listbox.get(tk.ACTIVE))
            self.listbox.destroy()
            self.listboxUp = False
            self.icursor(tk.END)

    def moveUp(self, event):
        if self.listboxUp:
            if self.listbox.curselection() == ():
                index = '0'
            else:
                index = self.listbox.curselection()[0]

            self.listbox.selection_clear(first=index)
            index = str(int(index) - 1)
            if int(index) == -1:
                index = str(self.listboxLength - 1)

            self.listbox.see(index)  # Scroll!
            self.listbox.selection_set(first=index)
            self.listbox.activate(index)

    def moveDown(self, event):
        if self.listboxUp:
            if self.listbox.curselection() == ():
                index = '-1'
            else:
                index = self.listbox.curselection()[0]

            if index != tk.END:
                self.listbox.selection_clear(first=index)
                if int(index) == self.listboxLength - 1:
                    index = "0"
                else:
                    index = str(int(index) + 1)

                self.listbox.see(index)  # Scroll!
                self.listbox.selection_set(first=index)
                self.listbox.activate(index)

    def comparison(self):
        return [w for w in self.autocompleteList if self.matchesFunction(self.var.get(), w)]


def matches(fieldValue, acListEntry):
    pattern = re.compile(re.escape(fieldValue) + '.*', re.IGNORECASE)
    return re.match(pattern, acListEntry)


root.iconbitmap('Images/corona.ico')

root.title("চিকিৎসায় রোগ নির্ণয়")

autocompleteList = list(main.trix.keys())

frame = LabelFrame(root, padx=10, pady=30, highlightthickness=2)
frame.pack(padx=50, pady=50)

c = Label(frame, text="চিকিৎসার ক্ষেত্রে রোগের পূর্বাভাস নির্ণয়", fg='blue4')
c.grid(row=0, column=0, columnspan=3, pady=(0, 20), padx=(30, 30), sticky="nsew")
c.config(font=("Consolas", 32, 'bold'))

L = Label(frame, text='প্রথম লক্ষন :')
L.grid(row=1, column=0, sticky=W, pady=(0, 10), padx=(120, 0))
L.config(font=("Consolas", 20))

p = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
p.grid(row=1, column=1)
p.config(font=('Consolas', 15, 'bold'))


def Destroy1():
    p.delete(0, END)


one = Label(frame, text='দ্বিতীয় লক্ষন :')
one.grid(row=2, column=0, sticky=W, pady=(0, 10), padx=(120, 0))
one.config(font=("Consolas", 20))
en = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
en.grid(row=2, column=1)
en.config(font=('Consolas', 15, 'bold'))


def Destroy2():
    en.delete(0, END)


aa = Label(frame, text='তৃতীয় লক্ষন :')
aa.grid(row=3, column=0, sticky=W, pady=(0, 10), padx=(120, 0))
aa.config(font=("Consolas", 20))

bb = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
bb.grid(row=3, column=1)
bb.config(font=('Consolas', 15, 'bold'))


def Destroy3():
    bb.delete(0, END)


dd = Label(frame, text='চতুর্থ লক্ষন :')
dd.grid(row=4, column=0, sticky=W, pady=(0, 10), padx=(120, 0))
dd.config(font=("Consolas", 20))

ee = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
ee.grid(row=4, column=1)
ee.config(font=('Consolas', 15, 'bold'))


def Destroy4():
    ee.delete(0, END)


gg = Label(frame, text='পঞ্চম লক্ষন :')
gg.grid(row=5, column=0, sticky=W, pady=(0, 10), padx=(120, 0))
gg.config(font=("Consolas", 20))

hh = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
hh.grid(row=5, column=1)
hh.config(font=('Consolas', 15, 'bold'))


def Destroy5():
    hh.delete(0, END)


def Destroy6():
    Destroy1()
    Destroy2()
    Destroy3()
    Destroy4()
    Destroy5()


kk = Button(frame, text='ফলাফল', command=main.prediction, bg='red', fg='white', activebackground='red')
kk.config(font=('Consolas', '18', 'bold'))
kk.grid(row=6, column=1, pady=(10, 50), padx=(0, 180))

jj = Button(frame, text=' পরিষ্কার ', bg='red', fg='white', activebackground='red', command=Destroy6)
jj.config(font=('Consolas', '18', 'bold'))
jj.grid(row=6, column=1, padx=(100, 0), pady=(10, 50), columnspan=2)

final_result = Entry(frame, width=50, borderwidth=0, bg='green', fg='white', justify=CENTER, insertbackground='green')
final_result.grid(row=8, column=0, pady=(0, 20), padx=(60, 0), columnspan=2)
final_result.config(font=('Consolas', 20, 'bold'))
final_result.bind("<Key>", lambda e: "break")

tt = Label(frame, text='বিঃদ্রঃ ভালো ফলাফল পেতে কমপক্ষে ৩ টি লক্ষন ব্যবহার করুন')
tt.grid(row=11, column=0, columnspan=2, padx=(60, 0))
tt.config(font=('', 15, 'bold'))

root.mainloop()
