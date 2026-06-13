from flask import Flask, jsonify, request,render_template
import yfinance as yf
import pandas as pd

app=Flask(__name__)


def get_detail(stock):

    

