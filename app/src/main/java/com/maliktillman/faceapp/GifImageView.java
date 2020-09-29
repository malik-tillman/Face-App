package com.maliktillman.faceapp;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Movie;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.view.View;

import java.io.InputStream;

public class GifImageView extends View {
    private Movie mMovie;
    private int mWidth, mHeight;
    private long mStart;

    private Context mContext;

    public GifImageView(Context context) {
        super(context);
        this.mContext = context;
    }

    public GifImageView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public GifImageView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        this.mContext = context;

        if (attrs.getAttributeName(1).equals("background")) {
            // Generate and set gif image id
            int id = Integer.parseInt(attrs.getAttributeValue(1).substring(1));

            // Open input stream to resource
            InputStream inputStream = mContext.getResources().openRawResource(id);

            // Make focusable
            setFocusable(true);

            // Initiate movie stream
            mMovie = Movie.decodeStream(inputStream);
            mWidth = mMovie.width();
            mHeight = mMovie.height();

            requestLayout();
        }
    }

    @Override protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        setMeasuredDimension(mWidth, mHeight);
    }

    @Override protected void onDraw(Canvas canvas) {
        long now = SystemClock.uptimeMillis();

        if (mStart == 0) {
            mStart = now;
        }

        if (mMovie != null) {

            int duration = mMovie.duration();
            if (duration == 0) {
                duration = 1000;
            }

            int relTime = (int) ((now - mStart) % duration);

            mMovie.setTime(relTime);

            mMovie.draw(canvas, 0, 0);
            invalidate();
        }
    }
}