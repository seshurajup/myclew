import React from 'react';
import {Composition} from 'remotion';
import {Short, ShortProps} from './Short';

const FPS = 30;

const demo: ShortProps = {
  code: 'def hello(name: str) -> str:\n    # fleet demo\n    return f"hi {name}"\n\nprint(hello("kaggle"))\n',
  language: 'python',
  title: 'hello.py',
  segments: [
    {start: 0, end: 2.5, text: 'Define a typed function'},
    {start: 2.5, end: 5, text: 'f-strings format inline'},
  ],
  cps: 30,
  tailSeconds: 2,
  maxSeconds: 90,
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="CodeShort"
      component={Short}
      width={1080}
      height={1920}
      fps={FPS}
      durationInFrames={300}
      defaultProps={demo}
      calculateMetadata={({props}) => {
        // duration = typing + transcript end + tail, capped at maxSeconds (YouTube hard limit 180s)
        const typing = props.code.length / Math.max(props.cps, 1);
        const lastSeg = props.segments.length
          ? Math.max(...props.segments.map((s) => s.end))
          : 0;
        const cap = Math.min(props.maxSeconds ?? 90, 180);
        const seconds = Math.min(cap, Math.max(typing, lastSeg) + props.tailSeconds);
        return {durationInFrames: Math.max(FPS, Math.round(seconds * FPS))};
      }}
    />
  );
};
